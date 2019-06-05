from __future__ import print_function
import chars2vec
import numpy as np
#from utils import *

# type: np.array(len(comments), dim), float32
# get comment embeddings with pretrained model
# models: eng_50, eng_100, eng_150, eng_200, eng_300
def getCommentEmbeddings(model_name, comments):
	if not model_name in ['eng_50', 'eng_100', 'eng_150', 'eng_200', 'eng_300']:
		print ("Error: arguments 'model' should be one of eng_50, eng_100, eng_150, eng_200, and eng_300")
		exit()
		
	if type(comments[0]) == str:
		c2v_model = chars2vec.load_model(model_name)
		comments = list(map(lambda x: x.strip(), comments))
		comment_embeddings = c2v_model.vectorize_words(comments)
		return comment_embeddings

	elif type(comments[0][0]) == str:
		comments_embeddings = []
		for comments_ in comments:
			c2v_model = chars2vec.load_model(model_name)
			comments_ = list(map(lambda x: x.strip(), comments_))
			comment_embeddings = c2v_model.vectorize_words(comments_)
			comments_embeddings.append(comment_embeddings)
		return comments_embeddings
		

# type: dict{str(ngram): int(count)}
# get ngram count dictionary
def getNgramCount(comments, n):
	ngram_count = {}
	for comment in comments:
		comment = comment.strip()
		for ngram_index in range(len(comment[:-n+1])):
			ngram = comment[ngram_index:ngram_index+n]
			if not ngram in ngram_count:
				ngram_count[ngram] = 0
			ngram_count[ngram] += 1

	return ngram_count

# type: list(str), list(int)
# get K most common ngrams and their counts
def getKMostCommonNgram(ngram_count, k=1):
	value = ngram_count.values()
	key = ngram_count.keys()

	kv = zip(key, value)
	kv = sorted(kv, reverse=True, key=lambda x:x[1])
	k_key = list(map(lambda x: x[0], kv[:k]))
	k_value = list(map(lambda x: x[1], kv[:k]))
	return k_key, k_value


# # type: np.array(dim), float32 // list(np.array(dim)), float32
# # get an embedding of comment, after aubtracting k most common ngram embeddings
# def getStylizedEmbeddings(model_name, comment, gramDict, dataset):
# 	n = len(gramDict.keys()[0])

# 	if type(comment) == str:
# 		common_ngram_embeddings = []

# 		comment = comment.strip()
# 		for ngram_index in range(len(comment[:-n+1])):
# 			ngram = comment[ngram_index:ngram_index+n]
# 			if ngram in gramDict:
# 				common_ngram_embeddings.append(gramDict[ngram])

# 		comment_embedding = getCommentEmbeddings(model_name, [comment])[0]

# 		for common_ngram_embedding in common_ngram_embeddings:
# 			comment_embedding = np.subtract(comment_embedding, common_ngram_embedding)
# 		return comment_embedding

# 	elif type(comment) == list and type(comment[0]) == list and type(comment[0][0]) == str:
# 		comment_embeddings = []
# 		for userComments in comment:
# 			avgUserComments = []
# 			for userComment in userComments:
# 				print (userComment)
# 				common_ngram_embeddings = []
# 				print(1)
# 				userComment = userComment.strip()
# 				for ngram_index in range(len(userComment[:-n+1])):
# 					ngram = userComment[ngram_index:ngram_index+n]
# 					if ngram in gramDict:
# 						common_ngram_embeddings.append(gramDict[ngram])
# 				print(2)
# 				comment_embedding = getCommentEmbeddings(model_name, [userComment])[0]
# 				print(3)
# 				if len(common_ngram_embeddings) > 0:
# 					for common_ngram_embedding in common_ngram_embeddings:
# 						comment_embedding = np.subtract(comment_embedding, common_ngram_embedding)
# 				print(4)
# 				avgUserComments.append(comment_embedding)
# 			dataset.setUserStyleEmbeddings(np.mean(avgUserComments, axis=0))

# 			raw_input() 
# 		return comment_embeddings

# 	else:
# 		print (type(comment))
# 		print (type(comment[0]))
# 		print (type(comment[0][0]))
# 		exit()