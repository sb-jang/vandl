from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from allennlp.modules.elmo import Elmo, batch_to_ids

data_dir = "./data/"
batch_size = 10
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False

# Image model part

def get_img_model():
	# Load the pretrained model
	img_model = models.resnet18(pretrained=True)
	set_parameter_requires_grad(img_model, True)

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
	img_feature = torch.zeros(512)
	img_feaure.to(device)

	def copy_data(m, i, o):
		img_feature.copy_(o.data)

	h = layer.register_forward_hook(copy_data)
	model(img_tensor)
	h.remove()

	return img_feature

# Post model part

"""
Input: tokenized post as a list of words
Output: ELMo representation (1, post_len, 1024)
"""
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
def get_post_feature(post):
	elmo = Elmo(options_file, weight_file, 1, dropout=0)
	character_ids = batch_to_ids(post)
	embedding = elmo(character_ids)

	return embedding

# Hashtag model part

class CharacterLevelCNN(nn.Module):
	def __init__(self, args):
		super(CharacterLevelCNN, self).__init__()

		# Load configuration such as model parameters
		with open(args.config_path) as f:
			self.config = json.load(f)

		# Creating conv layers
		conv_layers = []
		for i, conv_layer_parameter in enumerate(self.config['model_parameters'][args.size]['conv']):
			if i == 0: # first conv layer
				in_channels = args.number_of_characters + len(args.extra_characters)
				out_channels = conv_layer_parameter[0]
			else: # other conv layers
				in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]

			if conv_layer_parameter[2] != -1: # layers with pooling
				conv_layer = nn.Sequential(nn.Conv1d(in_channels, out_channels,
					kernel_size=conv_layer_parameter[1], padding=0),
				nn.ReLU(), nn.MaxPool1d(conv_layer_parameter[2]))
			else: # layers without pooling
				conv_layer = nn.Sequential(nn.Conv1d(in_channels, out_channels,
					kernel_size=conv_layer_parameter[1], padding=0),
				nn.ReLU())
			conv_layers.append(conv_layer)
		self.conv_layers = nn.ModuleList(conv_layers)

		input_shape = (args.batch_size, args_max_length,
			args.number_of_characters + len(args.extra_characters))
		dimension = self._get_conv_output(input_shape)

		print('dimension :', dimension)

		# Creating fc layers
		fc_layer_parameter = self.config['model_parameters'][args.size]['fc'][0]
		fc_layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(dimension, fc_layer_parameter), nn.Dropout(0.5)),
			nn.Sequential(
				nn.Linear(fc_layer_parameter, fc_layer_parameter), nn.Dropout(0.5)),
			nn.Linear(fc_layer_parameter, args_number_of_classes),])
		self.fc_layers = fc_layers

		if args.size == 'small':
			self._create_weights(mean=0.0, std=0.05)
		elif args.size == 'large':
			self._create_weights(mean=0.0, std=0.02)

	# Initialize weights for conv and fc layers
	def _create_weights(self, mean=0.0, std=0.05):
		for module in self.modules():
			if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
				module.weight.data.normal_(mean, std)

	def _get_conv_output(self, shape):
		input = torch.rand(shape)
		output= input.transpose(1, 2)
		# forward pass through conv layers
		for i in range(len(self.conv_layers)):
			output = self.conv_layers[i](output)

		output = output.view(output.size(0), -1)
		n_size = output.size(1)
		return n_size

	def forward(self, input):
		output = input.transpose(1, 2)
		# forward pass through conv layers
		for i in range(len(self.conv_layers)):
			output = self.conv_layers[i](output)

		output = output.view(output.size(0), -1)

		# forward pass through fc layers
		for i in range(len(self.fc_layers)):
			output = self.fc_layers[i](output)
		return output