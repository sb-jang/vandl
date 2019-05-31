import torch
from torch.utils.data import Dataset
from utils import *
import os
import numpy as np
class InstagramDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        self.root_path = './data/'

        for _, users_, _ in os.walk(self.root_path):
            self.users = users_
            break

        self.image_data = []
        self.comment_data = []
        self.post_data = []
        self.tag_data = []
        self.user_data = []

        self.imageFiles = {}
        for user in self.users:
            user_path = self.root_path + user
            for _, _, files in os.walk(user_path):
                imageFiles = list(filter(lambda x: '.jpg' in x or '.jpeg' in x, files))
                self.imageFiles[user] = list(imageFiles)
                for imageFile in imageFiles:
                    post = tagReplace(getPost(user, imageFile))
                    tags = getTags(user, imageFile)
                    comment = getUserComment(user, imageFile)
                    imageArray = getImageArray(user, imageFile, resize=True)

                    self.image_data.append(imageArray)
                    self.comment_data.append(comment)
                    self.post_data.append(post)
                    self.tag_data.append(tags)
                    self.user_data.append(user)

                    # print type(self.image_data[0])
                    # print type(self.comment_data[0])
                    # print type(self.post_data[0])
                    # print type(self.tag_data[0])
                    # print type(self.user_data[0])

                    # raw_input( )

        # print self.comment_data
        # raw_input()
        # print self.post_data
        # raw_input()
        # print self.tag_data
        # raw_input()
        self.num_users = self.users
        self.num_images = sum(map(lambda x: len(self.imageFiles[x]), self.users))
        self.num_data = len(self.comment_data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # print "dataloader.py"
        # print self.user_data[idx], type(self.user_data[idx])
        # print self.image_data[idx], type(self.image_data[idx])
        # print self.post_data[idx], type(self.post_data[idx])
        # print self.comment_data[idx], type(self.comment_data[idx][0])
        # print self.tag_data[idx], type(self.tag_data[idx][0])
        # print ''
        sample = {'user': self.user_data[idx], 'image': self.image_data[idx], 'post': self.post_data[idx], 'tags': self.tag_data[idx], 'comment': self.comment_data[idx]}
        return sample
        #return self.user_data[idx], self.image_data[idx], self.post_data[idx], self.tag_data[idx], self.comment_data[idx]