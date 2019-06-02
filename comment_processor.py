import chars2vec
import numpy as np
#from utils import *

# type: np.array(len(comments), dim), float32
# get comment embeddings with pretrained model
# models: eng_50, eng_100, eng_150, eng_200, eng_300
def getCommentEmbeddings(model_name, comments):
	if not model_name in ['eng_50', 'eng_100', 'eng_150', 'eng_200', 'eng_300']:
		print "Error: arguments 'model' should be one of eng_50, eng_100, eng_150, eng_200, and eng_300"

	c2v_model = chars2vec.load_model(model_name)
	comments = list(map(lambda x: x.strip(), comments))
	comment_embeddings = c2v_model.vectorize_words(comments)
	return comment_embeddings

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
	k_key = map(lambda x: x[0], kv[:k])
	k_value = map(lambda x: x[1], kv[:k])
	return k_key, k_value


# type: np.array(dim), float32
# get an embedding of comment, after aubtracting k most common ngram embeddings
def subtractCommonNgrams(model_name, comment, k_key):
	n = len(k_key[0])

	common_ngrams = []

	comment = comment.strip()
	for ngram_index in range(len(comment[:-n+1])):
		ngram = comment[ngram_index:ngram_index+n]
		if ngram in k_key:
			common_ngrams.append(ngram)

	common_embeddings = getCommentEmbeddings(model_name, common_ngrams)
	comment_embedding = getCommentEmbeddings(model_name, [comment])[0]

	for common_embedding in common_embeddings:
		comment_embedding = np.subtract(comment_embedding, common_embedding)

	return comment_embedding