import os
import numpy as np
from PIL import Image

# type: list(str)
# get a list of all users
def getAllUsers():
    path = './data'
    for _, users_, _ in os.walk(path):
        users = users_
        break
    return users

# type: list(str)
# get a list of all pictures where a user left comments
def getUserImageNames(user):
    path = './data/' + user
    for (_, _, fileList_) in os.walk(path):
        fileList = fileList_
    imageList = list(filter(lambda x: '.jpg' in x or '.jpeg' in x, fileList))
    return imageList

# type: list(str)
# get a list of all comments written on an image written by a user
def getUserComments(user, image):
    imagePath = './data/' + user + '/' + image
    assert os.path.exists(imagePath)

    if '.jpg' in image:
        shortCode = image.split('.jpg')[0]
    elif '.jpeg' in image:
        shortCode = image.split('.jpeg')[0]
    else:
        print("Error: not jpg or jpeg")
        exit()

    comments = []
    cnt = 0
    while os.path.exists('./data/' + user + '/' + shortCode + '.' + str(cnt)):
        commentPath = './data/' + user + '/' + shortCode + '.' + str(cnt)
        with open(commentPath, 'r') as f:
            line = eval(f.readline())
            comments.append(line[0])
        cnt += 1
    return comments

# type: (weight(int), height(int))
# get the size of an image
def getImageSize(user, image):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    print(im.size)

# type: None
# show the image
def showImage(user, image):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    im.show()

# type: np.array (width, height, rgb))
# get image as an array
def getImageArray(user, image):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    imageArray = np.array(im)
    return imageArray

# type: str or None
# get post with user and image name
def getPost(user, imagePath):
    if '.jpg' in imagePath:
        image = imagePath.split('.jpg')[0]
    elif '.jpeg' in imagePath:
        image = imagePath.split('.jpeg')[0]
    postPath = './data/' + user + '/' + image + '.text'
    if os.path.exists(postPath):
        with open(postPath, 'r') as f:
            line = f.readline()
            line = eval(line)
            assert len(line) == 1
            text = line[0]
    else:
        return None
    return text.strip()

# type: list(str)
# get a list of tags with user and image name
def getTags(user, imagePath):
    if '.jpg' in imagePath:
        image = imagePath.split('.jpg')[0]
    elif '.jpeg' in imagePath:
        image = imagePath.split('.jpeg')[0]
    tagPath = './data/' + user + '/' + image + '.tag'
    if os.path.exists(tagPath):
        with open(tagPath, 'r') as f:
            line = eval(f.readline())
            tags = line
    else:
        return None
    return tags

def tagReplace(comment):
    return ' '.join(map(lambda x: '@TAG' if x[0] == '@' else x, comment.split(' ')))