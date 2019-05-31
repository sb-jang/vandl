from __future__ import print_function
from comment_processor import *

comments = ["I love you", "He hates you"]

#print getCommentEmbeddings('eng_50', comments)
ngram_count = getNgramCount(comments, 2)
print (ngram_count)

most_ngram, _ = getKMostCommonNgram(ngram_count, 2)
print (most_ngram)

emb = subtractCommonNgrams('eng_50', "I love you", most_ngram)
print (emb)