from __future__ import print_function
import os
import numpy as np
from PIL import Image
import unicodedata

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

# type: str
# get the comment written on an image written by a user
def getComment(user, image):
    imagePath = './data/' + user + '/' + image
    assert os.path.exists(imagePath)

    if '.jpg' in image:
        shortCode = image.split('.jpg')[0]
    elif '.jpeg' in image:
        shortCode = image.split('.jpeg')[0]
    else:
        print ("Error: not jpg or jpeg")
        exit()

    #while os.path.exists('./data/' + user + '/' + shortCode + '.comment'):
    commentPath = './data/' + user + '/' + shortCode + '.0'
    with open(commentPath, 'r') as f:
        line = eval(f.readline())
        comment = unicode2str(line[0])
        #comment = line[0]
    return comment

# type: (weight(int), height(int))
# get the size of an image
def getImageSize(user, image):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    return im.size

# type: None
# show the image
def showImage(user, image):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    im.show()

# type: np.array (width, height, rgb))
# get image as an array
def getImageArray(user, image, resize=True):
    imagePath = './data/' + user + '/' + image
    im = Image.open(imagePath)
    imageArray = np.array(im)

    if resize:
        imageArray = np.array(Image.fromarray(imageArray).resize((224, 224), Image.ANTIALIAS))

    return imageArray

def unicode2str(uni):
    uni = unicodedata.normalize('NFKD', uni).encode('ascii', 'ignore')
    return str(uni)

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
            text = unicode2str(line[0])
            #text = line[0]

    else:
        return ''
    return text.strip()

# type: str
# get the tags with user and image name (delimiter: space)
def getTags(user, imagePath):
    if '.jpg' in imagePath:
        image = imagePath.split('.jpg')[0]
    elif '.jpeg' in imagePath:
        image = imagePath.split('.jpeg')[0]
    tagPath = './data/' + user + '/' + image + '.tag'
    if os.path.exists(tagPath):
        with open(tagPath, 'r') as f:
            line = eval(f.readline())
            if len(line) == 0:
                tags = ''
            else:
                tags = ' '.join(list(map(lambda x: str(x), line)))
                #tags = ' '.join(list(map(lambda x: x, line)))
    else:
        return ''
    return tags

def tagReplace(comment):
    if comment is '':
        return ''
    return ' '.join(map(lambda x: '@TAG' if len(x) > 0 and x[0] == '@' else x, comment.split(' ')))