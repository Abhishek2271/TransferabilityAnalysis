#!/usr/bin/env python
# File: cifar-convnet.py
# Author: Yuxin Wu
import argparse
import os
from tensorpack import tfv1 as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu

"""
A small convnet model for Cifar10.

A higher capacity version of Cifar10 base model defined in cifar_covnet.py and cifar_covnet_b.py

MODEL C CIFAR10

Cifar10 trained on 1 GPU:
    91% accuracy after 50k iterations.
    79 itr/s on P100
"""


class Model(ModelDesc):

    def inputs(self):
        return [tf.TensorSpec((None, 30, 30, 3), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        drop_rate = tf.constant(0.5 if self.training else 0.0)

        if self.training:
            tf.summary.image("train_image", image, 10)
        if tf.test.is_gpu_available():
            image = tf.transpose(image, [0, 3, 1, 2])
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

        #image = image / 4.0     # just to make range smaller
        #added conv2.3 and conv3.3
        with argscope(Conv2D, activation=BNReLU, use_bias=False, kernel_size=3), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format=data_format):
            logits = LinearWrap(image) \
                .Conv2D('conv1.1', filters=64) \
                .Conv2D('conv1.2', filters=64) \
                .MaxPooling('pool1', 3, stride=2, padding='SAME') \
                .Conv2D('conv2.1', filters=128) \
                .Conv2D('conv2.2', filters=128) \
                .Conv2D('conv2.3', filters=128) \
                .MaxPooling('pool2', 3, stride=2, padding='SAME') \
                .Conv2D('conv3.1', filters=128, padding='VALID') \
                .Conv2D('conv3.2', filters=128, padding='VALID') \
                .Conv2D('conv3.3', filters=128, padding='VALID') \
                .FullyConnected('fc0', 1024 + 512, activation=tf.nn.relu) \
                .Dropout(rate=drop_rate) \
                .FullyConnected('fc1', 512, activation=tf.nn.relu) \
                .FullyConnected('linear', 10)()

        tf.nn.softmax(logits, name="output")
        
        tf.add_to_collection("logits", logits)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)