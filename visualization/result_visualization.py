# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
Visualize the result compared to the standards.

Authors: AxsPlayer
Date: 2018/04/17 12:49:06
"""
import random
import pickle

import pandas as pd
import json

import matplotlib.pyplot as plt


class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database', 'version']
    PREDICTION_FIELDS = ['results', 'version']

    def __init__(self, ground_truth_filename=None,
                 frame_score=None,
                 video_id='113456300',
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 subset='testing'):

        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        self.ground_truth_filename = ground_truth_filename
        self.frame_score = frame_score
        self.video_id = video_id
        self.subset = subset
        self.gt_fields = ground_truth_fields
        # Import ground truth and predictions.
        self.ground_truth = self._import_ground_truth()

    def _import_ground_truth(self):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(self.ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].iteritems():
            if self.subset != v['subset']:
                continue
            for ann in v['annotations']:
                #                if ann['label'] not in activity_index:
                #                    activity_index[ann['label']] = cidx
                #                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(ann['segment'][0])
                t_end_lst.append(ann['segment'][1])
                #                label_lst.append(activity_index[ann['label']])
                label_lst.append(0)

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth

    def result_plot(self):

        print self.ground_truth[self.ground_truth['video-id'] == self.video_id]["t-end"].index
        video_data_index = self.ground_truth[self.ground_truth['video-id'] == self.video_id]["t-end"].index
        video_len = len(self.frame_score)
        t_end = self.ground_truth[self.ground_truth['video-id'] == self.video_id]["t-end"]
        t_start = self.ground_truth[self.ground_truth['video-id'] == self.video_id]["t-start"]

        ground_truth = [0] * video_len

        for index in video_data_index:
            interval = int(t_end[index]) - int(t_start[index])
            ground_truth[int(t_start[index]):int(t_end[index])] = [1] * interval

        plt.plot(ground_truth)
        plt.plot(self.frame_score)
        plt.show()


def plot(ground_truth_file=None,
         frame_score=None,
         video_id='113456300',
         subset='testing'):
    """
    Inputs
    -------
    ground_truth_file : str
        Full path to the ground truth json file.
    frame_score: list
        the score of each frame.
    video_id: str
        the id of the video

    Outputs
    -------
    plot the figure you want.

    """

    anet_detection = ANETdetection(ground_truth_file, frame_score, video_id=video_id,
                                   subset=subset)
    anet_detection.result_plot()


if __name__ == '__main__':

    cnt = 0
    with open('infer_res') as f:
        for line in f:
            splits = line.strip().split('\t')
            if len(splits) == 2:
                video_name = splits[0]
                scores = map(float, splits[1].split(','))
                cnt += 1
                if cnt == 2:
                    break
    # frame_score = [random.random()] * 2000

    plot(ground_truth_file="/mnt/BROAD-datasets/video/meta.json",
         frame_score=scores,
         video_id=video_name,
         subset='validation')

    with open('infer_res_final.pkl') as f:
        data = pickle.load(f)
    plot(ground_truth_file="/mnt/BROAD-datasets/video/meta.json",
         frame_score=data[video_name],
         video_id=video_name,
         subset='validation')