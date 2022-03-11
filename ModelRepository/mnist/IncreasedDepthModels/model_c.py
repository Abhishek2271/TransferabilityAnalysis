import tensorflow as tf

from tensorpack import *


"""
Define model capapcity (number of parameters a model has) for this model

This is mapped as below when giving input.

MODEL A = 32
MODEL B = 64 (100% of MODEL A)
MODEL C = 96 (200% of MODEL A)
"""
model_capacity = 48

class Model(ModelDesc):

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


    # Provide the input signature
    # Here it should be noted that the LeNet architecture that we are impmlementing requires images to be in 32x32 format
    # By default the image size is 28x28 for MNIST, so need to resize images before feeding to the network
    def inputs(self):
                #This is TF 1.13 code so we need to specify a placeholder for the graph and feed input to it later
        return [tf.TensorSpec((None, 28, 28), tf.float32, 'input'),
                # label is a Rank 1 tensor/ an array that holds true labels for the current minibatch and hence the (?) dimension
                # While building the grpah label placeholder is used to proovide the loss which the grpah returns. 
                # So when this grpah executes, it takes input and returns loss based on comparision with "output" and "label"
                # since the mini-batch is determined later during training
                # Label tensor is used only during training and is accessable later as "Label:0" tensor
                # Label is used only during training
                tf.TensorSpec((None,), tf.int32, 'label')]

    #define the model
    def build_graph(self, image, label):
        """
            The default dataset for MNIST only has 3 dim (Batch, image_height, Image_width). In tf, one addition dimension
            for channel is required so add one additional channel at axis =3

            Here also notics that input has same dimension as in inputs() TensorSpec meaning that data or input fed is in batches
            i.e input has dim: (128, 28, 28), 128 being the batch size and thus the accuracy computed laster is of 128 data points.
            tensorflow accepts input of (BHWC); Batch, height, width and channel 
        """
        image = tf.expand_dims(image, 3)

        #normalize image
        #image = image/255.0

        # Define the architecture.
        # Here we are using LeNet 5 architecture (http://yann.lecun.com/exdb/lenet/), known to have high accuracy for MNIST digit classifications
        print("is training ", self.training) 
        # conv2d: The default stride is (1,1) for tf.layers so not changing those
        # conv2d: The default padding is "same", default activation is "none" 
        # pooling: The default padding is "same", default stride is equal to the pool size
        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu, filters= model_capacity):
            # LinearWrap is just a syntax sugar.
            # See tutorial at https://tensorpack.readthedocs.io/tutorial/symbolic.html
            logits = (LinearWrap(image)
                      .Conv2D('conv0')
                      .Conv2D('conv0.1')
                      .Conv2D('conv0.2')
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv1')
                      .Conv2D('conv2')
                      .Conv2D('conv2.1')
                      .Conv2D('conv2.2')
                      .MaxPooling('pool1', 2)
                      .Conv2D('conv3')
                      .Conv2D('conv3.1')
                      .Conv2D('conv3.2')
                      .FullyConnected('fc0', 512, activation=tf.nn.relu)
                      .Dropout(rate=0.5 if self.training else 0.0)
                      .FullyConnected('linear', 10)())
        #using activation=tf.identity in last layer does not affect inference at all, similarily not adding condition to dropout,
        #during training also it should not affect

        # get the logits layer. Logits are final layer values which are not passed to softmax function. 
        # So basically tensors computed at last layer without any activation functions
        # while crafting adv examples logits are used because the softmax layer causes the output to lose its expressiveness        
        #logits = tf.layers.dense(l, 10, activation=tf.identity, name='linear')
        #finally pass the logits to the softmax to get the probabilities
        tf.nn.softmax(logits, name="output")
        tf.add_to_collection("logits", logits)

        # a vector of length B with loss of each sample
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')  # the average cross-entropy loss
        
        #for now do not regularize the weight
        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, loss], name='total_cost')
        summary.add_moving_summary(loss, wd_cost, total_cost)
        
        #summary.add_moving_summary(loss)

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error & accuracy (in a moving average fashion) This is different than the normal average. 
        # The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        # summary.add_param_summary(('.*/kernel', ['histogram', 'rms']))
        # weight and bias variables are named layerName/kernel (eg conv0/kernel) when using tf.layers in case of 
        # tensorpack's layer definition they are named layerName/W (eg conv0/W)
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost
        #return loss

    #define the optimizer
    def optimizer(self):
        return tf.train.AdamOptimizer()        