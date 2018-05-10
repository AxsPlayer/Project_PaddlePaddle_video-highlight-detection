# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
The main body to detect the target proposals and output results.

Authors: AxsPlayer
Date: 2018/05/10
"""
import numpy as np
import os
import cPickle
import json
import random
import gzip
import sys
import get_detection_performance as evalue
from matplotlib import pyplot as plt

import logging
import paddle.v2 as paddle
from paddle.v2.attr import ParamAttr

import train
import infer
import temporal_actionness_grouping


def main():
    # Set the logger, as well as arguments.
    logger = logging.getLogger("paddle")
    logger.setLevel(logging.INFO)
    args = {
        "use_gpu": True,
        "trainer_count": 1,
        "batch_size": 256,
        "num_batches_to_log": 10,
        "num_batches_to_test": 10,
        "num_batches_to_save_model": 20,
        "num_passes": 5,
        "model_path": None,
        "infer_res_path": "infer_res",
        "conv_config": [[2, 128], [3, 128], [4, 128]],
        "fc_config": [128, 64, 32],
    }

    # Initialize the PaddlePaddle.
    paddle.init(use_gpu=args['use_gpu'], trainer_count=args['trainer_count'])

    # Train the Neural Network.
    train()

    # Infer the result.
    args = {
        "use_gpu": True,
        "trainer_count": 1,
        "batch_size": 32,
        "num_batches_to_log": 10,
        "num_batches_to_test": 50,
        "num_batches_to_save_model": 20,
        "num_passes": 2,
        "model_path": "params_pass_npar8_20180226.tar",
        "infer_res_path": "infer_valid_20180228_1",
        "conv_config": [[3, 128], [5, 128], [7, 128]],
        "fc_config": [128, 64, 32],
    }
    infer()

    # Temporal actioness grouping.
    actioness_cutoff_floor, actioness_cutoff_ceiling, actioness_cutoff_step = 0.0, 1.0, 0.05
    maximum_portion_floor, maximum_portion_ceiling, maximum_portion_step = 0.0, 1.0, 0.05
    repeat_ratio = 0.95
    proposal_data = create_multiple_proposals(actioness_cutoff_ceiling, actioness_cutoff_floor, maximum_portion_ceiling,
                                 maximum_portion_floor, actioness_cutoff_step, maximum_portion_step, repeat_ratio)

    # Write the data into file.
    with open("train_test_v3.json", 'w') as fobj:
        json.dump(proposal_data, fobj)

    sys.path.append('/mnt/BROAD-datasets/video/eval_script/')

    # Evaluate the result to the ground truth.
    ground_truth = '/mnt/BROAD-datasets/video/meta.json'
    result_json = 'train_test_v3.json'
    subset = 'training'
    mAP = evalue.main(ground_truth, result_json, subset)
    res = {'name': 'mAP [0.5-0.95]', 'value': mAP}
    print res

    actioness_cutoff, maximum_portion = 0.79, 0.01

    proposal_data = {'results': {}, 'version': "VERSION 1.0"}

    # Write the test data into file.
    with open('infer_res.test') as f:
        for line in f:
            splits = line.strip().split('\t')
            video_name = splits[0]

            if len(video_name) > 9:
                continue

            scores = map(float, splits[1].split(','))
            video_length = len(scores)
            this_vid_proposals = []

            #         actioness_cutoff = plt.hist(scores)[1][-2]
            #         threshold = get_threshold(scores) - 0.05

            results = create_proposal(scores, actioness_cutoff, maximum_portion)
            for interval in results.values():
                if interval['end'] - interval['start'] < 50:
                    continue
                elif interval['end'] - interval['start'] > 420:
                    continue
                # elif not cal_over_ratio(scores[interval['start']: interval['end']], threshold, 0.35):
                #                 continue
                proposal = {
                    'score': interval['score'],
                    'segment': [interval['start'], interval['end']],
                }
                this_vid_proposals += [proposal]
            proposal_data['results'][video_name] = this_vid_proposals

    with open("test_20180313.json", 'w') as fobj:
        json.dump(proposal_data, fobj)


if __name__ == '__main__':
    main()