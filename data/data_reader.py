# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
The data reader to fetch features and tags.

Authors: AxsPlayer
Date: 2018/04/19 12:49:06
"""
# Import necessary package.
import cPickle
import gzip
import json
import logging
import numpy as np
import os
import random

import paddle.v2 as paddle
from paddle.v2.attr import ParamAttr


class Reader(object):
    """Data reader.

    Read train and test data according given path.

    """
    def __init__(self, n_frame=8, frame_fea_size=2048, pos_prob=0.241, neg_prob=0.1,
                 label_file="/mnt/BROAD-datasets/video/meta.json",
                 train_data_image_path="/mnt/BROAD-datasets/video/training/image_resnet50_feature/",
                 train_data_audio_path="/mnt/BROAD-datasets/video/training/audio_feature/",
                 valid_data_image_path="/mnt/BROAD-datasets/video/validation/image_resnet50_feature/",
                 valid_data_audio_path="/mnt/BROAD-datasets/video/validation/audio_feature/",
                 test_data_image_path='/mnt/BROAD-datasets/video/testing/image_resnet50_feature/'):
        """

        :param n_frame:
        :param frame_fea_size:
        :param pos_prob:
        :param neg_prob:
        :param label_file:
        :param train_data_image_path:
        :param train_data_audio_path:
        :param valid_data_image_path:
        :param valid_data_audio_path:
        :param test_data_image_path:
        """
        self.n_frame = n_frame
        self.frame_fea_size = frame_fea_size
        self.pos_prob = pos_prob
        self.neg_prob = neg_prob
        self.label_file = label_file
        self.train_data_image_path = train_data_image_path
        self.train_data_audio_path = train_data_audio_path
        self.valid_data_image_path = valid_data_image_path
        self.valid_data_audio_path = valid_data_audio_path
        self.test_data_image_path = test_data_image_path
        self.test_data = self.train_data

    @staticmethod
    def load_json(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            return data

    @staticmethod
    def get_video_names(path):
        return set(os.listdir(path))

    def get_label(self, filename, path, subset="training"):
        # todo add audio names if there are some errors.
        video_names = self.get_video_names(path)
        json_data = self.load_json(filename)
        database = json_data['database']
        train_dict = {}
        for video_name in database.keys():
            video_info = database[video_name]
            video_subset = video_info["subset"]
            if video_subset == subset and video_name + '.pkl' in video_names:
                train_dict[video_name] = video_info['annotations']
        return train_dict

    def train_data(self):
        train_label = self.get_label(self.label_file, self.train_data_image_path)

        def gen_data():
            for video_name in train_label:
                with open(self.train_data_image_path + str(video_name) + ".pkl", 'rb') as f:
                    img_feat = cPickle.load(f)
                # with open(self.train_data_audio_path + str(video_name) + ".pkl", 'rb') as f:
                #   audio_feat = cPickle.load(f)
                img_len = len(img_feat)
                # audio_len = len(audio_feat)
                labels = [0] * img_len
                for inter in train_label[video_name]:
                    idx0, idx1 = inter["segment"]
                    idx0 = int(idx0)
                    idx1 = int(idx1)
                    labels[idx0: idx1] = [1] * (idx1 - idx0)
                window_extend_size = [self.n_frame / 2, self.n_frame - self.n_frame / 2 - 1]
                window_prefix = np.tile(img_feat[0, :], (window_extend_size[0], 1))
                window_suffix = np.tile(img_feat[-1, :], (window_extend_size[1], 1))
                window_img = np.row_stack((window_prefix, img_feat, window_suffix))
                f = open('logs', 'a')
                print >> f, "Processing video: %s" % video_name
                f.close()
                for i in range(img_len):
                    if labels[i] == 1 and random.random() <= self.pos_prob:
                        yield (window_img[i:i + self.n_frame, :], labels[i])
                    elif labels[i] == 0 and random.random() <= self.neg_prob:
                        yield (window_img[i:i + self.n_frame, :], labels[i])
        return gen_data

    def test_data(self):
        test_label = self.get_label(self.label_file, self.valid_data_image_path, "validation")

        def gen_data():
            for video_name in train_label:
                with open(self.train_data_path + str(video_name) + ".pkl", 'rb') as f:
                    img_feat = cPickle.load(f)
                img_len = len(img_feat)
                labels = [0] * img_len
                for inter in train_label[video_name]:
                    idx0, idx1 = inter["segment"]
                    idx0 = int(idx0)
                    idx1 = int(idx1)
                    labels[idx0: idx1] = [1] * (idx1 - idx0)
                window_extend_size = [self.n_frame / 2, self.n_frame - self.n_frame / 2 - 1]
                window_prefix = np.tile(img_feat[0, :], (window_extend_size[0], 1))
                window_suffix = np.tile(img_feat[-1, :], (window_extend_size[1], 1))
                window_img = np.row_stack((window_prefix, img_feat, window_suffix))
                # window_audio = np.row_stack((window_prefix, audio_feat, window_suffix))
                for i in range(img_len):
                    if labels[i] == 1 and random.random() <= self.pos_prob:
                        # yield ({'img_feature': window_img[i:i + self.n_frame, :],
                        #      'audio_feature': window_audio[i:i + self.n_frame, :]}, labels[i])
                        yield (window_img[i:i + self.n_frame, :], labels[i])  # window_audio[i:i + self.n_frame, :],
                    elif labels[i] == 0 and random.random() <= self.neg_prob:
                        # yield ({'img_feature': window_img[i:i + self.n_frame, :],
                        #      'audio_feature': window_audio[i:i + self.n_frame, :]}, labels[i])
                        yield (window_img[i:i + self.n_frame, :], labels[i])  # window_audio[i:i + self.n_frame, :],

        return gen_data

    def result_train_data(self):
        valid_label = self.get_label(self.label_file, self.train_data_image_path, "training")

        for video_name in valid_label:
            with open(self.train_data_image_path + str(video_name) + ".pkl", 'rb') as f:
                img_feat = cPickle.load(f)
            # with open(self.valid_data_audio_path + str(video_name) + ".pkl", 'rb') as f:
            #   audio_feat = cPickle.load(f)
            yield (img_feat, video_name)  # audio_feat,

    def result_video_data(self):
        valid_label = self.get_label(self.label_file, self.valid_data_image_path, "validation")

        for video_name in valid_label:
            with open(self.valid_data_image_path + str(video_name) + ".pkl", 'rb') as f:
                img_feat = cPickle.load(f)
            # with open(self.valid_data_audio_path + str(video_name) + ".pkl", 'rb') as f:
            #   audio_feat = cPickle.load(f)
            yield (img_feat, video_name)  # audio_feat,