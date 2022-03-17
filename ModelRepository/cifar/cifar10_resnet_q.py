#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.dataflow import imgaug
from tensorpack.tfutils.varreplace import remap_variables
from dorefa import get_dorefa

BITW = 2 # Bitwidth of weight
BITA = 2 # Bitwidth of activations
BITG = 32 #Bitwidth of gradients

'''
NOTE: 
THIS IS NOT AN ORIGIAL WORK FROM ME.

THIS IS THE EXACT IMPLEMENTATION OF RESNET IMPLEMENTED BY AUTHORS OF TENSORPACK. THE ORIGINAL CODE AT.
https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py


THE DATA FED TO THE NETWORK IS CHANGED AS TO ACCOMODATE THE OBJECTIVE AND CONSISTENCY WITH OTHER MODELS BEING COMPARED
IN THE THESIS


The DoReFa-Net implementation is from: https://github.com/tensorpack/tensorpack/pull/69
But this method does not quite work for 2-bit or more quantization, added an additional BN layer on second conv layer as a quick-fix. 
This does not seem to hamper the working but decreases training time significantly

COMMENT FROM AUTHORS:


'''
"""
CIFAR10 ResNet example. Reproduce the 2-GPU settings in:
"Deep Residual Learning for Image Recognition", with following exceptions:
* This implementation uses the architecture variant proposed in:
  "Identity Mappings in Deep Residual Networks"
* This model uses the whole training set instead of a train-val split.

Results:
* ResNet-110(n=18): about 5.9% val error after 64k steps (8.3 step/s)

To train:
    ./cifar10-resnet.py --gpu 0,1
"""

# paper uses 2 GPU with a total batch size of 128
BATCH_SIZE = 128  # per-gpu batch size


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        #image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)


        def quantize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'linear' in name:
                return v
            else:
                logger.info("Quantizing weights to {} bits {}".format(BITW, v.op.name)) #v.name = conv1/W:0; v.op.name = conv1/W
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            logger.info("Quantizing activations to {} bits {}".format(BITA, x.name))
            return fa(nonlin(x))

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=BNReLU)
                c1 = activate(c1)
                #need to use BNReLU here since without it the network accuracy is 56%
                c2 = Conv2D('conv2', c1, out_channel, activation=BNReLU) 
                c2 = activate(c2)#activate(c2)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l


        with remap_variables(quantize_weight), \
            argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
            argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, activation=BNReLU)
            l = activate(l)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

            logits = FullyConnected('linear', l, 10)

        tf.nn.softmax(logits, name='output')
        tf.add_to_collection("logits", logits)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        # monitor training accuracy
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt     
