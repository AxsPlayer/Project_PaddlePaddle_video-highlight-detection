# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
Infer the data result using trained Nerual Network.

Authors: AxsPlayer
Date: 2018/04/17 12:49:06
"""
import numpy as np
import os
import cPickle
import json
import random
import gzip

import logging
import paddle.v2 as paddle
from paddle.v2.attr import ParamAttr

import data_reader
import network_structure

# Set the logger, as well as arguments.
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def infer(model_path=args['model_path'],
          infer_res_path=args['infer_res_path'],
          conv_config=args['conv_config'],
          fc_config=args['fc_config']):
    # prepare infer data
    def _infer_reader():
        data_reader = Reader()
        slice_size = data_reader.n_frame
        video_extend_size = [slice_size / 2, slice_size - slice_size / 2 - 1]
        for video, video_name in data_reader.train_data():
            n_video_frames = len(video)
            video_prefix = np.tile(video[0, :], (video_extend_size[0], 1))
            video_suffix = np.tile(video[-1, :], (video_extend_size[1], 1))
            video = np.row_stack((video_prefix, video, video_suffix))
            for i in range(n_video_frames):
                video_slice = video[i:i + slice_size, :]
                yield video_slice, video_name, i
    infer_batchs = paddle.batch(_infer_reader, 1000)

    # create inferer
    data_reader = Reader()
    frame_fea_size = data_reader.frame_fea_size
    cost, prediction, label = Network(
        conv_config=conv_config, fc_config=fc_config, input_len=frame_fea_size)()
    logger.info("Load the trained model from [%s]." % model_path)
    parameters = paddle.parameters.Parameters.from_tar(
        open(model_path, "r"))
    inferer = paddle.inference.Inference(
        output_layer=prediction, parameters=parameters)

    # infer
    logger.warning("Write infer result to [%s]." % infer_res_path)
    feeding = {"frame_slice": 0}
    last_video = ""
    score_list = []
    with open(infer_res_path, "w") as f:
        for batch in infer_batchs():
            logger.info("processing video:%s, frame:%s" % (batch[0][1], batch[0][2]))
            res = inferer.infer(input=batch, feeding=feeding)
            assert len(res) == len(batch), ("error show during infer, plz check!")
            for frame_input, frame_res in zip(batch, res):
                video = frame_input[1]
                score_res = str(frame_res[1])
                if video != last_video:
                    if len(score_list) != 0:
                        line = '\t'.join([last_video, ','.join(score_list)]) + '\n'
                        f.write(line)
                    score_list = []
                    last_video = video
                score_list.append(score_res)
        line = '\t'.join([last_video, ','.join(score_list)]) + '\n'
        f.write(line)
