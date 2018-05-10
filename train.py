# !/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018 Personal. All Rights Reserved
#
################################################################################
"""
The process to train Neural Network.

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


def train(batch_size=args['batch_size'],
          num_batches_to_log=args['num_batches_to_log'],
          num_batches_to_test=args['num_batches_to_test'],
          num_batches_to_save_model=args['num_batches_to_save_model'],
          num_passes=args['num_passes'],
          conv_config=args['conv_config'],
          fc_config=args['fc_config'], ):
    data_reader = Reader()
    train_data = data_reader.train_data()
    test_data = data_reader.test_data()
    frame_fea_size = data_reader.frame_fea_size
    train_batchs = paddle.batch(
        paddle.reader.shuffle(train_data, buf_size=1000),
        batch_size=batch_size)

    test_batchs = paddle.batch(
        paddle.reader.shuffle(test_data, buf_size=1000),
        batch_size=batch_size)

    cost, prediction, label = Network(
        conv_config=conv_config, fc_config=fc_config, input_len=frame_fea_size)()

    parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-4,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    model_trainer = paddle.trainer.SGD(
        cost=cost,
        extra_layers=paddle.evaluator.auc(input=prediction, label=label),
        parameters=parameters,
        update_equation=adam_optimizer)

    feeding = {
        "frame_slice": 0,
        "label": 1
    }

    def _event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            # output train log
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)

                #             # test model
                #             if event.batch_id > 0 and event.batch_id % num_batches_to_test == 0:
                #                 result = model_trainer.test(reader=test_batchs, feeding=feeding)
                #                 print "\nPass %d, Batch %d, %s" % (
                #                     event.pass_id, event.batch_id, result.metrics)
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            print "\nModel saved, params_pass_%d.tar" % event.pass_id

    def old_event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            # output train log
            if event.batch_id % num_batches_to_log == 0:
                logger.info("Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))

            # test model
            if event.batch_id > 0 and \
                                    event.batch_id % num_batches_to_test == 0:
                result = model_trainer.test(reader=test_batchs, feeding=feeding)
                logger.info("Test at Pass %d, %s" % (event.pass_id,
                                                     result.metrics))
            # save model
            if isinstance(event, paddle.event.EndPass):
                # if event.batch_id > 0 and \
                #         event.batch_id % num_batches_to_save_model == 0:
                args['model_path'] = "passid{passid}_batchid{batchid}.tar.gz".format(
                    passid=str(event.pass_id), batchid=str(event.batch_id))
                logger.info("save model into [%s]" % (args['model_path']))
                with gzip.open(args['model_path'], 'w') as f:
                    parameters.to_tar(f)

    model_trainer.train(
        reader=train_batchs,
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

    logger.info("Training has finished.")