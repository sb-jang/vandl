from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torchvision import models, transforms
from allennlp.modules.elmo import Elmo, batch_to_ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
options_file = "./elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "./elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

# Image model part

def get_img_model():
	# Load the pretrained model
	img_model = models.resnet18(pretrained=True)
	set_parameter_requires_grad(img_model, True)
	img_model = img_model.to(device)

	# Use the model object to select the desired layer
	layer = img_model._modules.get('avgpool')

	# Set model to evaluation mode
	img_model.eval()

	return img_model, layer

"""
Input: normalized image tensor (3,224,224)
Output: feature vector of the image (512)
"""
def get_img_feature(img_tensor, model, layer):
	img_tensor = img_tensor.unsqueeze(0)
	# The 'avgpool' layer has an output size of 512
	img_feature = torch.zeros(1, 512, 1, 1)
	img_feature = img_feature.to(device)

	def copy_data(m, i, o):
		img_feature.copy_(o.data)

	h = layer.register_forward_hook(copy_data)
	model(img_tensor)
	h.remove()

	return img_feature.squeeze()

# Post model part

def get_post_model():
	model = Elmo(options_file, weight_file, 1, dropout=0)
	return model
"""
Input: tokenized post as a list of words
Output: ELMo representation (1, post_len, 1024)
"""
def get_post_feature(post, model):
	character_ids = batch_to_ids(post)
	embedding = model(character_ids)

	return embedding['elmo_representations'][0]

# img_transforms = transforms.Compose([
# 		transforms.Resize(224),
# 		transforms.ToTensor(),
# 		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 		])
def get_embedding(image, post, tags, img_embedder, img_layer, post_embedder, char_embedder):
	# Extract image feature
	img = torch.tensor(image, dtype=torch.float32, device=device)
	img = img.permute(2, 0, 1)
	img_feature = get_img_feature(img, img_embedder, img_layer)

	# Extract post feature
	if post != '':
		tokenized_post = post.split(' ')
		post_feature = get_post_feature(tokenized_post, post_embedder)
		post_feature = torch.mean(post_feature, dim=1).squeeze(1)
		post_feature = torch.mean(post_feature, dim=0).squeeze(0)
		post_feature = post_feature.to(device)
	else:
		post_feature = torch.zeros(1024, device=device)

	# Extract hashtag feature
	if tags != '':
		tokenized_tags = tags.split(' ')
		hashtag_feature = char_embedder.vectorize_words(tokenized_tags)
		hashtag_feature = torch.mean(torch.tensor(hashtag_feature, device=device), dim=0).squeeze(0)
	else:
		hashtag_feature = torch.zeros(50, device=device)

	return (img_feature, post_feature, hashtag_feature)

# seq2seq model part

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Encoder, self).__init__()
		self.input_size = sum(input_size)
		self.hidden_size = hidden_size

		# self.fc_img = nn.Linear(input_size[0], hidden_size)
		# self.fc_post = nn.Linear(input_size[1], hidden_size)
		# self.fc_hash = nn.Linear(input_size[2], hidden_size)
		self.fc = nn.Linear(self.input_size, self.hidden_size)

	def forward(self, input):
		input = torch.cat((input[0], input[1], input[2]), 0)
		embedded = input.view(1, 1, -1)
		output = embedded
		output = self.fc(output)
		output = F.relu(output)
		return output

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

MAX_LENGTH = 15
class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_output):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)