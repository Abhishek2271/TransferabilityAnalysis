import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.varreplace import remap_variables

from dorefa import get_dorefa


BITW = 2 # Bitwidth of weight
BITA = 2 # Bitwidth of activations
BITG = 32 #Bitwidth of gradients
Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"

"""
    BIAS QUANTIZATION: 
    Biases are not quantized by DoReFa because according to authors they contribute less in network size and performance so 
    quantizing them is not helpful for performance or for size reduction. 
    Ref: pending (is in github issues responses)
    GRADIENT QUANTIZATION:
    DoReFaNet supports gradient quantization. In fact, one of its achievement is being able to quantize gradients to 8 bits and lower
    and still have comparable accuracy (atleast for 8 bits)
    The main advantage of gradient quantization is that it allows bitwise operations during backpropagation as well where kernels
    can be operated with gradients without float point multiplication but instead use bitwise operation. This significantly boosts
    training speeds further, transfering gradients from one network to another during parallel training is faster when bitwidth is smaller.
    ref: 
        https://arxiv.org/pdf/1606.06160.pdf, section 1 and 2.1
        https://arxiv.org/pdf/1808.04752.pdf, section 4.3
    Gradient quantization is not necessary for our research since we do not need to take into account performance boost during training.    
"""

"""
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
"""

class Model(ModelDesc):

    # Provide the input signature
    # Here it should be noted that the LeNet architecture that we are impmlementing requires images to be in 32x32 format
    # By default the image size is 28x28 for MNIST, so need to resize images before feeding to the network
    def inputs(self):
        return [tf.TensorSpec((None, 28, 28), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    #define the model
    def build_graph(self, image, label):
        """
            The default dataset for MNIST only has 3 dim (Batch, image_height, Image_width). In tf, one addition dimension
            for channel is required so add one additional channel at axis =3
        """
        image = tf.expand_dims(image, 3)

        # get quantization functions from dorefa
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

        #normalize image
        #input = input/255.0

        # Define the architecture.
        # Here we are using LeNet 5 architecture (http://yann.lecun.com/exdb/lenet/), known to have high accuracy for MNIST digit classifications
        with remap_variables(quantize_weight): #quantize weights            
            # although weights are not quantized in the first layer to preserve accuracy, activation map from first layer are quantized
            # ref.section 2.7 https://arxiv.org/pdf/1606.06160.pdf
            # The layers are represented in symbolic input format:
            # ref: 
            #   https://tensorpack.readthedocs.io/tutorial/symbolic.html
            # This representation is same as tf.layers but we are using this format because the linearwrap()
            # provides apply() function that allows to apply function directly to a tensor            
            # ref. 
            #   https://tensorpack.readthedocs.io/en/latest/modules/models.html#tensorpack.models.Conv2D
            #   https://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/models/linearwrap.html#LinearWrap
            
            # conv2d: The default stride is (1,1) for tf.layers so not changing those  
            logits = (LinearWrap(image)
                        .Conv2D('conv0',6, 5, padding = "same", activation= tf.nn.tanh)
                        .AvgPooling('pool0', 2, 2, padding='valid').apply(activate) 
                        .Conv2D('conv1',16, 5, padding = "valid", activation= tf.nn.tanh)
                        .AvgPooling('pool1', 2, 2, padding='valid').apply(activate)                  
                        .FullyConnected('fc0', 120, activation=tf.nn.tanh).apply(activate)
                        .FullyConnected('fc1',84, activation=tf.nn.tanh)
                        .FullyConnected('linear', 10)())
            #can use tf.layers for quantization too but they are very slow so using the   symbolic method provided by tensorpack
            """
            l = tf.layers.conv2d(input, 6, 5, padding = "same", activation= tf.nn.tanh, name='conv0')
            l = tf.layers.average_pooling2d(l, 2, 2, padding='valid')           
            l = tf.map_fn(activate, l)              
            #l = tf.vectorized_map(activate, l)                   
            l = tf.layers.conv2d(l, 16, 5, padding = "valid", activation= tf.nn.tanh, name='conv1')
            l = tf.layers.average_pooling2d(l, 2, 2, padding='valid')
            l = tf.map_fn(activate, l)     
            l = tf.layers.flatten(l)
            l = tf.layers.dense(l, 120, activation=tf.nn.tanh, name='fc0')
            l = tf.map_fn(activate, l)     
            l = tf.layers.dense(l, 84, activation=tf.nn.tanh, name='fc1')
            before_quantization = tf.identity(l, name='b_quant')
            l = tf.map_fn(activate, l)     
            after_quantization = tf.identity(l, name='a_quant')"""
               
            # get the logits layer. Logits are final layer values which are not passed to softmax function. 
            # So basically tensors computed at last layer without any activation functions
            # while crafting adv examples ART uses the logits because the softmax layer causes the output to lose its expressiveness        
            #logits = tf.layers.dense(l, 10, activation=tf.identity, name='linear')
        tf.nn.softmax(logits, name="output")
        #logits_1 = tf.identity(logits, name='quantized')
        #print(logits_1)
        tf.add_to_collection("logits", logits) 

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
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return loss

    #define the optimizer
    def optimizer(self):
        return tf.train.AdamOptimizer()
