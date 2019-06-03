import shutil
import os
import random

train_ratio = 0.8
val_ratio = 0.2


root_path = './data/'

train_dir = './train_data'
val_dir = './val_data'


if os.path.exists(train_dir):
    os.rmdir(train_dir)
if os.path.exists(val_dir):
    os.rmdir(val_dir)

os.mkdir(train_dir)
os.mkdir(val_dir)


for _, users_, _ in os.walk(root_path):
    users = users_
    break
#self.users = list(filter(lambda x: 'a' <= x[0] and x[0] <= 'e', self.users))
image_data = []
comment_data = []
post_data = []
tag_data = []
user_data = []
user_comments = {}
user_style_embedding = {}

imageFiles = {}
print ("Reading user information")
cnt_ = 0

for user in users:
    user_path = root_path + user
    os.mkdir(val_dir + '/' + user)
    os.mkdir(train_dir + '/' + user)
    for _, _, files in os.walk(user_path):
        imageFiles = list(filter(lambda x: '.jpg' in x or '.jpeg' in x, files))
        imageFiles = list(map(lambda x: x.split('.jpg')[0], imageFiles))
        imageFiles = list(map(lambda x: x.split('.jpeg')[0], imageFiles))

        random.shuffle(imageFiles)

        num_val = len(imageFiles) * val_ratio
        num_train = len(imageFiles) - num_val

        for cnt, imageFile in enumerate(imageFiles):
            post = user_path + '/' + imageFile + '.text'
            tags = user_path + '/' + imageFile + '.tag'
            comment = user_path + '/' + imageFile + '.0'
            if os.path.exists(user_path + '/' + imageFile + '.jpg'):
                image_ = user_path + '/' + imageFile + '.jpg'
            else:
                image_ = user_path + '/' + imageFile + '.jpeg'

            if cnt < num_val:
                dest_dir = val_dir
            else:
                dest_dir = train_dir

            if os.path.exists(post):
                shutil.copy(post, dest_dir + '/' + user + '/')
            if os.path.exists(tags):
                shutil.copy(tags, dest_dir + '/' + user + '/')
            if os.path.exists(comment):
                shutil.copy(comment, dest_dir + '/' + user + '/')
            if os.path.exists(image_):
                shutil.copy(image_, dest_dir + '/' + user + '/')
            
            # print type(self.image_data[0])
            # print type(self.comment_data[0])
            # print type(self.post_data[0])
            # print type(self.tag_data[0])
            # print type(self.user_data[0])

            # raw_input( )
    cnt_ += 1
    if int(float(cnt_) / len(users) * 100) % 10 == 0:
        print (str(int(float(cnt_) / len(users) * 100)) + '% reading completed')
