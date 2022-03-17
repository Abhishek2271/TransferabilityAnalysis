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
from tensorpack.tfutils.varreplace import remap_variables
from dorefa import get_dorefa

"""
THIS IS THE REIMPLEMENTAITON OF CODE FROM:
https://github.com/tensorpack/tensorpack/blob/master/examples/basics/cifar-convnet.py
With addition of quantization layers.
The problem here is  that for higher bitwidth quantization (> 2-bit) the network takes too long to train.
To fix this added a BN layer in the final conv layer which seems to fix the issue withtout any other repercussions


A small convnet model for Cifar10.

This is a quantized version of the CIFAR model described in cifar_covnet_b.py (base cifar model)

CIFAR QUANTIZED MODEL B


Cifar10 trained on 1 GPU:
    91% accuracy after 50k iterations.
    79 itr/s on P100
"""

from dorefa import get_dorefa

BITW = 2 # Bitwidth of weight
BITA = 2 # Bitwidth of activations
BITG = 32 #Bitwidth of gradients


class Model(ModelDesc):

    def inputs(self):
        return [tf.TensorSpec((None, 30, 30, 3), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):       

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


        if self.training:
            tf.summary.image("train_image", image, 10)
        if tf.test.is_gpu_available():
            image = tf.transpose(image, [0, 3, 1, 2])
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

        with remap_variables(quantize_weight), \
                argscope(Conv2D, activation=BNReLU, use_bias=False, kernel_size=3), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format=data_format):
            logits = (LinearWrap(image) 
                .Conv2D('conv1.1', filters=64).apply(activate)
                .Conv2D('conv1.2', filters=64) 
                .MaxPooling('pool1', 3, stride=2, padding='SAME').apply(activate)
                .Conv2D('conv2.1', filters=128).apply(activate) 
                .Conv2D('conv2.2', filters=128) 
                .MaxPooling('pool2', 3, stride=2, padding='SAME').apply(activate) 
                .Conv2D('conv3.1', filters=128, padding='VALID').apply(activate) 
                .Conv2D('conv3.2', filters=128, padding='VALID').apply(activate)
                .FullyConnected('fc0', 1024 + 512, activation=tf.nn.relu).apply(activate) 
                .Dropout(rate=0.5 if self.training else 0.0)
                .FullyConnected('fc1', 512, activation=tf.nn.relu) 
                .FullyConnected('linear', 10)())

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
