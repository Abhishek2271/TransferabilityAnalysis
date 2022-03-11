#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu

import argparse
import os
#from tkinter import _Padding
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.dataflow import imgaug
from DataSets.cifar import Cifar10, get_cifar10_data
Cifar_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\CIFAR10"

'''

NOTE: 
THIS IS NOT AN ORIGIAL WORK FROM ME.

THIS IS THE EXACT IMPLEMENTATION OF RESNET IMPLEMENTED BY AUTHORS OF TENSORPACK. THE ORIGINAL CODE AT.
https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py


THE DATA FED TO THE NETWORK IS CHANGED AS TO ACCOMODATE THE OBJECTIVE AND CONSISTENCY WITH OTHER MODELS BEING COMPARED
IN THE THESIS

COMMENT FROM AUTHORS:

https://github.com/tensorpack/tensorpack/pull/69 DOREFA IMPLEMENTATION
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

class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        #removed image scaling for now since the net performs similarily without it.
        #image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])
    
        # A residual block with two convolution blocks. with conv1 followed by a BN layer
        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
                #pad = "valid"
            else:
                out_channel = in_channel
                stride1 = 1
                #pad = "same"
            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)
                #when stride is 2 same padding will reduce the dimension by half
                c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                #when increase_dim = true in 2.0 and 3.0, two things happen
                # 1. Channel dimesion is increased (say from 16,32,32) to (32,32,32) because both conv1 and conv2 
                #    use increase_dim
                # 2. The stride = 2 in conv1 will then reduce the dimension of the image from say (32,32,32) to (32,16,16)
                # Thus for C2+l at the end of the residual block, we then need to first do pooling so that l2 image dim is reduces.
                # we then need to pad l so that the channel dim is increased for l
                if increase_dim:
                    # Note that the pooling is done for the layer "l" which is the input to this residual block, 
                    # this makes sense because c2 will decrease the dimension of tensor from say (32,32,32) to (32,16,16) when increase dim=true
                    # so need to do pooling so that c2 and l can be later added                    
                    l = AvgPooling('pool', l, 2)
                    #padding is done because when increase_dim is "true" c2 dimension is increased by double so pad the image similarily
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l

        #Here, in the thesis report, pooling layer is not mentioned in the "model destails" apart from Global pooling
        #in the the end because pooling within layers is done to the input to the resnet block so as to make the dim equal 
        # add it to the last block. So it is not pooling for feature extraction in a sense. So have skipped it.
        
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, activation=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16 (results in (?, 16, 32, 32) tensor

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32 (results in (?, 32, 16, 16) tensor

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64 (results in (?, 64, 8, 8) tensor)
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, 10)
        tf.nn.softmax(logits, name='output')
        tf.add_to_collection("logits", logits)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_vector')
        # monitor training 
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        # monitor training accuracy
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        #Adapt regularization hyperparam.
        #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        #                       = 0.0002 * 0.2(globalstep/48000)
        #Decay weights every 480000 str by a base of 0.2
        # Since staircase decay is true the division of (global_step/decay_steps) is int so till 480000 steps
        # weight decay param is 0.0002*(0.2^0). At 480000 and after 480000, till 959,000 decay is 0.2*0.0002 and at 960,000
        # the decay is 0.0002*(0.2^2) and so on.

        #In our case this huge number of steps is never reached so the weight decay param is 0.0002
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        #this is simply multiplying wd_w with sigma(w^2). So wd_cost = lambda.(sigma(w^2))
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt