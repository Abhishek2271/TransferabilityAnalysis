import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

from tensorpack import *
from tensorpack.callbacks import graph
from tensorpack.dataflow import dataset
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from tensorpack.input_source import PlaceholderInput
from tensorpack.tfutils.common import get_tensors_by_names, get_op_tensor_name
from tensorpack.tfutils.tower import PredictTowerContext
#if i call this, this will call the updated version of the file in my PC but without it tensorpack import * will call the defaule version somehow
#from tensorpack.predict.base import OfflinePredictor

from art.estimators.classification import TensorFlowClassifier

#setup all parameters for creating attack
from .setup_attacks import SetupAttacks

def create_classifier(attack_params):

    '''
    Create classifier for initiating the attack
    
    args:
    ---------------
        attack_params: Contains parameters required create a whitebox attack classifier.
        
    returns:
    --------------------
    
        An ART classifier that can be used to create adversarial example 
    '''
    #graph = tf.get_default_graph()
    with attack_params.current_graph.as_default():    #the graph should be same as created when restoring the model otherwise the an empty graph is created without any variables
        print("--------------Creating adversarial examples.-----")
        # using the sparse_softmax_cross_entropy here and using the loss tensor from network does not make any difference.
        #loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_tensors))
        #need to assign new name for the adamoptimizer as the layers the variables with similar name already exists in the model during training
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='MyNewAdam')
        #seems trainer is not necessary when creating attacks 
        # https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/estimators/classification/tensorflow.py
        train = optimizer.minimize(attack_params.loss)
        max_= 1.0
        min_= 0.0
        classifier = TensorFlowClassifier(
                clip_values=(min_, max_),
                input_ph=attack_params.input_placeholder[0], #input placeholder
                output=attack_params.logits,   #this is basically the whole network info as the logitsare the final layer (without softmax)
                                            #so the TFClassifier uses the logits instead of output from the softmax, "output" in the model for creating adversarial examples
                labels_ph= attack_params.output_placeholder[0],  #label/ correct placeholder for label for the data_set to train, labels` must have the dtype of `int32` or `int64`.
                #train=train,
                loss=attack_params.loss,
                learning=None,
                sess=attack_params.sess,
                preprocessing_defences=[],
                ) 
        return classifier


def RestoreModel(config):

    """
    Restore the model to first get all necessary attack parameters.
    This function is re-implementation of OfflinePredictor of tensorpack with necessary changes
    
    args:
    ----------
    
    config: The prediction configuration containing 
        1. The model description
        2. Input signature
        3. Output tensor (not signature) from model
        4. Session with data to restore checkpoint (checkpoint location is in the session which is used to restore the graph here)
        
    2 and 3 are really not required but they are by default in the offline predictor but not really useful here since the output tensors 
    used here is the output placeholder than the actual output(actual output is used during inference)
    """

    #setup the graph
    # This part is a custom OfflinePrediction made for creating adversarial samples
    graph = config._maybe_create_graph()
    with graph.as_default():
            input = PlaceholderInput()
            input.setup(config.input_signature)
            print("PREPARING THE TOWER FUNCTION ....")
            with PredictTowerContext(''):
                config.tower_func(*input.get_input_tensors())
            print("TOWER FUNCTION PREPARED....")
            input_tensors = get_tensors_by_names(config.input_names) #This gives the input placeholder/signature
            #output_tensors = graph.get_tensor_by_name("label:0")    #This gives the output placeholder/signature
            #output tensor here might be a misnomer. This is a "label". 
            #   It just shows what the shape of the label is like if it is a single value (like in MNIST/CIFAR) or can be matrix
            output_tensors = get_tensors_by_names(["label"])
            output_net = get_tensors_by_names(config.output_names)
            print("output_shape:", output_net[0].shape)
            print("input tensor:", input_tensors[0])
            print("output_tensor:", output_net[0])
            #print(output_tensors.shape)
            #print(output_label[0].shape)
            # logits are variables before they are passed ot the softmax function. 
            # Here they represent an entire model because it called functions from previous layers to get the final value
            print(len(tf.get_collection("logits")))
            print(tf.get_collection("logits")) #get collection returns an array. Since "logits" has only one element we take the first element
            logits = tf.get_collection("logits")[0]  
            #taking softmax or logits it does not really make a difference in output(emperical obeservation)
            #  Since we are using softmax_cross_entropy there is no need to get the softmax again so use the logits
            #  DONT USE SOFTMAX
            out_prob = tf.nn.softmax(logits, name="output")              
            print("these are the logits:", logits)   
            config.session_init._setup_graph()
            sess = config.session_creator.create_session()
            #print("values in logits", sess.run(logits))
            print("RESTORING GRAPH...")
            config.session_init._run_init(sess)
            print("GRAPH RESTORED...")
            loss = graph.get_tensor_by_name("cross_entropy_loss:0") 
            attack_params = SetupAttacks(sess, input_tensors, output_tensors, logits, loss, graph)
            return attack_params