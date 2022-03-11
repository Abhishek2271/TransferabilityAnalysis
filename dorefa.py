# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu

import tensorflow as tf
import sys


def get_dorefa(bitW, bitA, bitG):
    """
    Return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    """
    #G = tf.get_default_graph()
    def quantize(x, k):
        n = float(2 ** k - 1)

        #C:\Python37\Lib\site-packages\tensorflow\python\ops\custom_gradient.py        
        @tf.custom_gradient
        #custom_gradients provide a more efficient or numerically stable gradient for a sequence of operations

        #tf.custom_gradients are used only during backward passees and during forward pass normal values of _quantize(x) are used
        #in backward pass, to compute gradients gWbk (from paper) and gabk are used i.e quantized versions of activations and 
        # #weights are used this why theuy should be passed during gradient computation. 
        def _quantize(x):
            #here lambda dy:dy enables some kind of iteration when _quantize(x) is called inside tf.custom_gradient.
            #since tf.custom-gradient is a decorator, with _quantize(x) we are calling custom_gradient with _quantize as an argument
            return tf.round(x * n) / n, lambda dy: dy
        #get weight and activations and return a quantized tensor
        #print(_quantize(x))
        #print("type", type(_quantize(x)))
        #here _quantize calls custom_gradient with "_quantize" as an argument. The x then is passed as an argument to the 
        #wrapper function inside custom_gradient because "_quantize" calls custom_gardient(funtion) then "_quantize()"" calls
        #custom_gradient(function) which returns a function then the () in then end will again execute the function with x as 
        #argument
        return _quantize(x)

    def fw(x):
        if bitW == 32:
            
            return x

        if bitW == 1:   # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
            print("quant weight")
            @tf.custom_gradient
            def _sign(x):
                return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy
            print("aaa", _sign(x))
            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        #tf.print(2 * quantize(x, bitW) - 1, output_stream=sys.stderr)
        print("aaa", 2 * quantize(x, bitW) - 1)
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    def fg(x):        
        #if bitG == 32:
        return x
        #Do not quantize gradients
        """
        @tf.custom_gradient
        def _identity(input):
            def grad_fg(x):
                rank = x.get_shape().ndims
                assert rank is not None
                maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
                x = x / maxx
                n = float(2**bitG - 1)
                x = x * 0.5 + 0.5 + tf.random_uniform(
                    tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
                x = tf.clip_by_value(x, 0.0, 1.0)
                x = quantize(x, bitG) - 0.5
                return x * maxx * 2

            return input, grad_fg
        return _identity(x)"""

    return fw, fa, fg