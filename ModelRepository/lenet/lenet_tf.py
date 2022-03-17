import tensorflow as tf
from tensorpack import *

'''
A simple lenet architecture. Implemented from paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
This type of layer definition without using the Tensorpack layer definition causes problems when adding quantization layers.
calling activation and weight function from between is very slow. This is however is know issue with Tensorflow and is fixed in later versions.

'''


Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"

'''
    Define model using the ModelDesc parent. To train or to inference a network, you would require 
    1. The dataset (to train or infer from)
    2. The model definition
    3. The input placeholder which is used to later feed data
    4. Optimizer function (required by trainer only)

    Three of these four (2,3,4) are supplied by the ModelDesc. Thus this class the core of tensorpack understanding and a
    model should be restored (especially quantized) using the ModelDesc description in order to make sure the graph is made correctly
    ref: 
        1. https://tensorpack.readthedocs.io/en/latest/tutorial/inference.html#step-1-build-the-model-graph
        2. https://tensorpack.readthedocs.io/tutorial/training-interface.html#with-modeldesc-and-trainconfig
'''
class Model(ModelDesc):

    # Provide the input signature
    # Here it should be noted that the LeNet architecture that we are impmlementing requires images to be in 32x32 format
    # By default the image size is 28x28 for MNIST, so need to resize images before feeding to the network
    def inputs(self):
        return [tf.TensorSpec((None, 28, 28), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    #define the model
    def build_graph(self, input, label):
        """
            The default dataset for MNIST only has 3 dim (Batch, image_height, Image_width). In tf, one addition dimension
            for channel is required so add one additional channel at axis =3
        """
        input = tf.expand_dims(input, 3)
        print("input shape is,", input.shape)
        print("label shape is,", label.shape)
        #normalize image
        #input = input/255.0

        # Define the architecture.
        # Here we are using LeNet 5 architecture (http://yann.lecun.com/exdb/lenet/), known to have high accuracy for MNIST digit classifications
         
        # conv2d: The default stride is (1,1) for tf.layers so not changing those
        l = tf.layers.conv2d(input, 6, 5, padding = "same", activation= tf.nn.tanh, name='conv0')
        l = tf.layers.average_pooling2d(l, 2, 2, padding='valid')
        l = tf.layers.conv2d(l, 16, 5, padding = "valid", activation= tf.nn.tanh, name='conv1')
        l = tf.layers.average_pooling2d(l, 2, 2, padding='valid')
        l = tf.layers.flatten(l)
        l = tf.layers.dense(l, 120, activation=tf.nn.tanh, name='fc0')
        l = tf.layers.dense(l, 84, activation=tf.nn.tanh, name='fc1')
        # get the logits layer. Logits are final layer values which are not passed to softmax function. 
        # So basically tensors computed at last layer without any activation functions
        # while crafting adv examples logits are used because the softmax layer causes the output to lose its expressiveness        
        logits = tf.layers.dense(l, 10, activation=tf.identity, name='linear')
        tf.nn.softmax(logits, name="output")
        tf.add_to_collection("logits", logits)

        print("logits shape is:", logits.shape)
        # a vector of length B with loss of each sample
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        # summary.add_param_summary(('.*/kernel', ['histogram', 'rms']))
        # weight and bias variables are named layerName/kernel (eg conv0/kernel) when using tf.layers in case of 
        # tensorpack's layer definition they are named layerName/W (eg conv0/W)
        summary.add_param_summary(('.*/kernel', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return loss

    #define the optimizer
    def optimizer(self):
        return tf.train.AdamOptimizer()
