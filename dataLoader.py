from __future__ import print_function

import torch
from torch.utils.data import Dataset
from utils import *
import os
import numpy as np
import time
from comment_processor import *

class InstagramDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, phase, num_users=None):
        assert phase == 'train' or phase == 'val' or phase == 'all'

        if phase == 'all':
            self.root_path = './data/'
        elif phase == 'train':
            self.root_path = './train_data/'
        elif phase == 'val':
            self.root_path = './val_data/'

        start = time.time()
        for _, users_, _ in os.walk(self.root_path):
            if num_users is not None:
                self.users = users_[:num_users]
            else:
                self.users = users_
            break
        #self.users = list(filter(lambda x: 'a' <= x[0] and x[0] <= 'e', self.users))
        self.image_data = []
        self.comment_data = []
        self.post_data = []
        self.tag_data = []
        self.user_data = []
        self.user_comments = {}
        self.user_style_embedding = {}

        self.imageFiles = {}
        print ("Reading user information")
        cnt = 0
        for user in self.users:
            user_path = self.root_path + user
            for _, _, files in os.walk(user_path):
                imageFiles = list(filter(lambda x: '.jpg' in x or '.jpeg' in x, files))
                self.imageFiles[user] = list(imageFiles)
                for imageFile in imageFiles:
                    post = tagReplace(getPost(user, imageFile))
                    tags = getTags(user, imageFile)
                    comment = tagReplace(getComment(user, imageFile))
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
            cnt += 1

            if int(float(cnt) / len(self.users) * 100) % 10 == 0:
                print (str(int(float(cnt) / len(self.users) * 100)) + '% reading completed')
        # print self.comment_data
        # raw_input()
        # print self.post_data
        # raw_input()
        # print self.tag_data
        # raw_input()
        self.num_users = self.users
        self.num_images = sum(map(lambda x: len(self.imageFiles[x]), self.users))
        self.num_data = len(self.comment_data)

        for user in self.users:
            self.user_comments[user] = []
            imageFiles = self.imageFiles[user]
            commentFiles = list(map(lambda x: x.replace('.jpg', '.0'), imageFiles))
            commentFiles = list(map(lambda x: x.replace('.jpeg', '.0'), commentFiles))
            for commentFile in commentFiles:
                with open(self.root_path + user + '/' + commentFile, 'r') as f:
                    line = eval(f.readline())
                    comment = unicode2str(line[0]).lower()
                    self.user_comments[user].append(comment)
        print ("Load data: " + str(time.time() - start))

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

    def getAllUsersComments(self, users):
        all_comments = []
        for user in users:
            # print (user)
            assert user in self.user_comments
            all_comments.append(self.user_comments[user])
        return all_comments

    def getUserStyleEmbedding(self, user, model_name, gramDict):
        n = len(list(gramDict.keys())[0])
        emb_path = './embeddingCache/' + model_name + '/'
        if os.path.exists(emb_path):
            if type(user) == str:
                avgUserComments = []
                embFile = emb_path + user + '.emb'
                assert os.path.exists(embFile)

                with open(embFile, 'r') as f:
                    while True:
                        common_ngram_embeddings = []
                        userComment = f.readline()
                        emb = f.readline()

                        if not userComment: break
                        emb = eval(emb)
                        for ngram_index in range(len(userComment[:-n+1])):
                            ngram = userComment[ngram_index:ngram_index+n]
                            if ngram in gramDict:
                                common_ngram_embeddings.append(gramDict[ngram])

                        comment_embedding = emb
                        if len(common_ngram_embeddings) > 0:
                            for common_ngram_embedding in common_ngram_embeddings:
                                comment_embedding = np.subtract(comment_embedding, common_ngram_embedding)
                        avgUserComments.append(comment_embedding)
                return np.mean(avgUserComments, axis=0)

            elif type(user) == list and type(user[0]) == str:
                embeddings = []
                for user_ in user:
                    avgUserComments = []
                    embFile = emb_path + user_ + '.emb'
                    assert os.path.exists(embFile)

                    with open(embFile, 'r') as f:
                        while True:
                            common_ngram_embeddings = []
                            userComment = f.readline()
                            emb = f.readline()
                            if not userComment: break

                            emb = eval(emb)
                            for ngram_index in range(len(userComment[:-n+1])):
                                ngram = userComment[ngram_index:ngram_index+n]
                                if ngram in gramDict:
                                    common_ngram_embeddings.append(gramDict[ngram])

                            comment_embedding = emb
                            if len(common_ngram_embeddings) > 0:
                                for common_ngram_embedding in common_ngram_embeddings:
                                    comment_embedding = np.subtract(comment_embedding, common_ngram_embedding)
                            avgUserComments.append(comment_embedding)
                    embeddings.append(np.mean(avgUserComments, axis=0))
                return embeddings

            else:
                print("Error")

        else:
            print ("Embeddings for " + model_name + " are not cached.")
            exit()
            # if type(user) == str:
            #     if not user in self.user_style_embedding or not n in self.user_style_embedding[user]:
            #         self.setUserStyleEmbedding(user, model_name, gramDict)
            #     return self.user_style_embedding[user][n]

            # elif type(user) == list and type(user[0]) == str:
            #     embeddings = []
            #     for user_ in user:
            #         if not user_ in self.user_style_embedding or not n in self.user_style_embedding[user_]:
            #             self.setUserStyleEmbedding(user_, model_name, gramDict)
            #         embeddings.append(self.user_style_embedding[user_][n])
            #     return embeddings

            # else:
            #     print("Error")
            #     exit(0)

    def setUserStyleEmbedding(self, user, model_name, gramDict):
        n = len(gramDict.keys()[0])

        if not user in self.user_style_embedding:
            self.user_style_embedding[user] = {}

        userComments = self.getAllUsersComments([user])[0]

        avgUserComments = []
        for userComment in userComments:
            common_ngram_embeddings = []
            userComment = userComment.strip()
            if len(userComment) == 0:
                continue
            for ngram_index in range(len(userComment[:-n+1])):
                ngram = userComment[ngram_index:ngram_index+n]
                if ngram in gramDict:
                    common_ngram_embeddings.append(gramDict[ngram])

            comment_embedding = getCommentEmbeddings(model_name, [userComment])[0]
            if len(common_ngram_embeddings) > 0:
                for common_ngram_embedding in common_ngram_embeddings:
                    comment_embedding = np.subtract(comment_embedding, common_ngram_embedding)
            avgUserComments.append(comment_embedding)
        self.user_style_embedding[user][n] = np.mean(avgUserComments, axis=0)

    def saveUserCommentEmbedding(self, user, model_name):
        print ("Saving " + user)
        userComments = self.getAllUsersComments([user])[0]
        if len(userComments) == 0:
            return
        userEmbeddings = getCommentEmbeddings(model_name, userComments)

        with open("./embeddingCache/" + user + ".emb", 'w') as f:
            for c, e in zip(userComments, userEmbeddings):
                f.write(str(c))
                f.write('\n')
                f.write(str(e.tolist()))
                f.write('\n')

    # def savePostEmbedding(self):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #     posts = list(set(self.post_data))

    #     options_file = "./elmo_2x4096_512_2048cnn_2xhighway_options.json"
    #     weight_file = "./elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    #     cnt = 0
    #     with open("./postCache.emb", 'w') as f:
    #         for post in posts:
    #             if post != '':
    #                 tokenized_post = nltk.word_tokenize(post.lower())

    #                 elmo = Elmo(options_file, weight_file, 1, dropout=0)
    #                 character_ids = batch_to_ids(tokenized_post)
    #                 embedding = elmo(character_ids)

    #                 post_feature = embedding['elmo_representations'][0]
    #                 post_feature = torch.mean(post_feature, dim=1).squeeze(1)
    #                 post_feature = torch.mean(post_feature, dim=0).squeeze(0)
    #                 post_feature = post_feature.to(device)
    #                 f.write(str(post))
    #                 f.write('\n')
    #                 f.write(str(post_feature.tolist()))
    #                 f.write('\n')
    #             cnt += 1
    #             print (cnt / len(posts))

