from __future__ import print_function

from comment_processor import *
from dataLoader import *
from torch.utils.data import DataLoader


dataset = InstagramDataset('train')
allComments = dataset.getAllUsersComments(dataset.users)
allComments = [y for x in allComments for y in x]
_2gramCount = getNgramCount(allComments, 2)
_3gramCount = getNgramCount(allComments, 3)
_4gramCount = getNgramCount(allComments, 4)
_5gramCount = getNgramCount(allComments, 5)


k = 10
modelName = 'eng_50'


_kMostCommon2gram, _2gramValues = getKMostCommonNgram(_2gramCount, k)
_kMostCommon3gram, _3gramValues = getKMostCommonNgram(_3gramCount, k)
_kMostCommon4gram, _4gramValues = getKMostCommonNgram(_4gramCount, k)
_kMostCommon5gram, _5gramValues = getKMostCommonNgram(_5gramCount, k)

_kMostCommon2gramEmbeddings = getCommentEmbeddings(modelName, _kMostCommon2gram)
_kMostCommon3gramEmbeddings = getCommentEmbeddings(modelName, _kMostCommon3gram)
_kMostCommon4gramEmbeddings = getCommentEmbeddings(modelName, _kMostCommon4gram)
_kMostCommon5gramEmbeddings = getCommentEmbeddings(modelName, _kMostCommon5gram)

_2gramDict = {}
_3gramDict = {}
_4gramDict = {}
_5gramDict = {}

for gram, emb in zip(_kMostCommon2gram, _kMostCommon2gramEmbeddings):
	_2gramDict[gram] = emb
for gram, emb in zip(_kMostCommon3gram, _kMostCommon3gramEmbeddings):
	_3gramDict[gram] = emb
for gram, emb in zip(_kMostCommon4gram, _kMostCommon4gramEmbeddings):
	_4gramDict[gram] = emb
for gram, emb in zip(_kMostCommon5gram, _kMostCommon5gramEmbeddings):
	_5gramDict[gram] = emb

train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
for d in train_loader:
	print (d['user'])  # user
	print (d['image']) # image array
	print (d['post'])  # post
	print (d['tags'])  # tags (string, delimiter=space)
	print (d['comment']) # comment

	# user style embedding: average of (user comment - common n-grams)'s
	print (dataset.getUserStyleEmbedding(d['user'], modelName, _5gramDict))
	print (dataset.getUserStyleEmbedding(d['user'][0], modelName, _5gramDict))

	input()