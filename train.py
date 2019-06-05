from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time, math, os, random
import copy
import utils, model, dataLoader
from torch.utils.data import DataLoader
import chars2vec
import nltk
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = dataLoader.InstagramDataset('train')
val_data = dataLoader.InstagramDataset('val')
train_data_size = train_data.__len__()
val_data_size = val_data.__len__()

img_model, img_layer = model.get_img_model()
c2v_model = chars2vec.load_model('eng_50')
input_size = [512, 1024, 50]

learning_rate = 0.001
hidden_size = 256
MAX_LENGTH = 15
batch_size = 100

SOS_token = 0
EOS_token = 1

class Vocab:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2 # Count SOS and EOS

	def addSentence(self, sentence):
		for word in nltk.word_tokenize(sentence):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

# Preparing training data
def prepareData(data):
	vocab = Vocab('comments')
	print("Reading data...")
	data_loader = DataLoader(dataset=data, batch_size=train_data_size, shuffle=False)
	for d in data_loader:
		print("Read %d sentences" % len(d['comment']))
		print("Counting words...")
		for c in d['comment']:
			vocab.addSentence(c)
		print("Counted words: %d" % vocab.n_words)

	return vocab

vocab = prepareData(train_data)

def indexesFromSentence(vocab, sentence):
	return [vocab.word2index[word] for word in nltk.word_tokenize(sentence)]

def tensorFromSentence(vocab, sentence):
	indexes = indexesFromSentence(vocab, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Train
teacher_forcing_ratio = 0.5

def train(input_tensors, target_tensor, encoder, decoder, encoder_optimizer,
	decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	target_length = target_tensor.size(0)

	encoder_output = torch.zeros(1, encoder.hidden_size, device=device)

	loss = 0

	encoder_output = encoder(input_tensors)

	decoder_input = torch.tensor([[SOS_token]], device=device)
	decoder_hidden = encoder_output

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			loss += criterion(decoder_output, target_tensor[di])
			decoder_input = target_tensor[di]
	else:
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()

			loss += criterion(decoder_output, target_tensor[di])
			if decoder_input.item() == EOS_token:
				break

	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length
"""
Helper functions to print time elapsed and estimated time remaining
given the current time and progress %
"""
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, print_every=10, plot_every=10, learning_rate=learning_rate):
	start = time.time()
	plot_losses = []
	print_loss_total = 0 # Reset every print_every
	plot_loss_total = 0 # Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()

	train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

	n_iters = train_data_size // batch_size
	cnt = 0
	for d in train_loader:
		# pdb.set_trace()
		for i in range(len(d['image'])):
			input_tensors = model.get_embedding(d['image'][i], d['post'][i], d['tags'][i],
				img_model, img_layer, c2v_model)
			target_tensor = tensorFromSentence(vocab, d['comment'][i])

			loss = train(input_tensors, target_tensor, encoder, decoder,
				encoder_optimizer, decoder_optimizer, criterion)
			print_loss_total += loss
			plot_loss_total += loss

		cnt += 1
		print("Training completed for %d batches." % cnt)

		if cnt % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, cnt / n_iters),
				cnt, cnt / n_iters * 100, print_loss_avg))

		if cnt % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	showPlot(plot_losses)

plt.switch_backend('agg')

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)

# Evaluate
def evaluate(encoder, decoder, image, post, tag, max_length=MAX_LENGTH):
	with torch.no_grad():
		input_tensors = model.get_embedding(image, post, tag,
			img_model, img_layer, c2v_model)

		encoder_output = torch.zeros(1, encoder.hidden_size, device=device)
		encoder_output = encoder(input_tensors)

		decoder_input = torch.tensor([[SOS_token]], device=device)
		decoder_hidden = encoder_output

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(vocab.index2word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_words

def evaluateRandomly(encoder, decoder, n=20):
	val_loader = DataLoader(dataset=val_data, batch_size=n, shuffle=True)
	for d in val_loader:
		for i in range(n):
			output_words = evaluate(encoder, decoder,
				d['image'][i], d['post'][i], d['tags'][i])
			output_sentence = ' '.join(output_words)
			print('ground truth:', d['comment'][i])
			print('generated:', output_sentence)
		break

encoder = model.Encoder(input_size, hidden_size).to(device)
decoder = model.DecoderRNN(hidden_size, vocab.n_words).to(device)

trainIters(encoder, decoder, learning_rate=learning_rate)
evaluateRandomly(encoder, decoder, 30)
