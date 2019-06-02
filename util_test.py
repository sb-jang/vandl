import numpy as np

from utils import *
# getAllUsers()
# getUserImageNames(user)
# getUserComments(user, image)
# getImageSize(user, image)
# showImage(user, image)
# getImageArray(user, image)
# getPost(user, imagePath)
# getTags(user, imagePath)
# tagReplace(comment)

ids = {}

users = getAllUsers()
image = getUserImageNames(users[1])[10]
arr = getImageArray(users[1], image)
#showImage(users[1], image)
#print arr.shape

comments = getUserComments(users[1], image)
for comment in comments:
    print tagReplace(comment)
print users[1]
post = getPost(users[1], image)
print tagReplace(post)
tags = getTags(users[1], image)
print tags


