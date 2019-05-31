from __future__ import print_function

comments = ["I love you", "He hates you"]

from comment_processor import *
from dataLoader import *
from torch.utils.data import DataLoader

#print getCommentEmbeddings('eng_50', comments)
#ngram_count = getNgramCount(comments, 2)
#print ngram_count 

#most_ngram, _ = getKMostCommonNgram(ngram_count, 2)
#print most_ngram

#emb = subtractCommonNgrams('eng_50', "I love you", most_ngram)
#print emb


dataset = InstagramDataset()
print (dataset.__len__())
#print dataset[0]

train_loader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
for d in train_loader:
	print (d['user'])  # user
	print (d['image']) # image array
	print (d['post'])  # post
	print (d['tags'])  # tags (string, delimiter=space)
	print (d['comment']) # comment
	print (dataset.getAllUsersComments(d['user']))
	raw_input()
