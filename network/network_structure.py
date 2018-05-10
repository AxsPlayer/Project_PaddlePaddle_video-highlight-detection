# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
Define the structure of Neural Network.

Authors: AxsPlayer
Date: 2018/04/19 12:49:06
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


class Network(object):
    def __init__(self, conv_config=[[2, 128], [3, 128], [4, 128]],
                 fc_config=[128, 64, 32], input_len=2048, classnum=2):
        self.conv_config = conv_config
        self.fc_config = fc_config
        self.input_len = input_len
        self.classnum = classnum

    def __call__(self):
        return self._build_model()

    def _build_model(self):
        frame_slice = paddle.layer.data(
            name="frame_slice",
            type=paddle.data_type.dense_vector_sequence(self.input_len))
        label = paddle.layer.data(
            name="label",
            type=paddle.data_type.integer_value(self.classnum))
        mid_layer = self._create_cnn(frame_slice)
        dnn = self._create_dnn(mid_layer)
        prediction = paddle.layer.fc(input=dnn,
                                     size=self.classnum,
                                     act=paddle.activation.Softmax())
        cost = paddle.layer.classification_cost(input=prediction, label=label)
        return cost, prediction, label

    def _create_cnn(self, input_layer):
        def create_conv(context_len, hidden_size, prefix):
            key = "%s_%d_%d" % (prefix, context_len, hidden_size)
            conv = paddle.networks.sequence_conv_pool(
                input=input_layer,
                context_len=context_len,
                hidden_size=hidden_size,
                # set parameter attr for parameter sharing
                context_proj_param_attr=ParamAttr(name=key + "contex_proj.w"),
                fc_param_attr=ParamAttr(name=key + "_fc.w"),
                fc_bias_attr=ParamAttr(name=key + "_fc.b"),
                pool_bias_attr=ParamAttr(name=key + "_pool.b"))
            return conv

        conv_layers = []
        for cfg in self.conv_config:
            logger.info(
                "create conv_layer of which context %s, filter number %s" % (cfg[0], cfg[1]))
            cur_conv = create_conv(cfg[0], cfg[1], "cnn")
            conv_layers.append(cur_conv)

        return paddle.layer.concat(input=conv_layers)

    def _create_dnn(self, input_layer):
        for id, dim in enumerate(self.fc_config):
            name = "fc_%d_%d" % (id, dim)
            logger.info("create fc_layer which dimention is %d" % dim)
            fc = paddle.layer.fc(
                input=input_layer,
                size=dim,
                act=paddle.activation.Tanh(),
                param_attr=ParamAttr(name="%s.w" % name),
                bias_attr=ParamAttr(name="%s.b" % name, initial_std=0.))
            input_layer = fc
        return input_layer
