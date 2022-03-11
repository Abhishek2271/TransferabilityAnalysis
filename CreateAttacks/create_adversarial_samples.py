from typing import Dict
from .attack_core import RestoreModel, create_classifier
from .setup_attacks import SetupAttacks
from .supported_attacks import SupportedAlgorithms
from tensorpack.utils import logger
import visualize_data 
import logging

import numpy as np
import pandas as pd

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import BoundaryAttack
from art.attacks.evasion import UniversalPerturbation
from art.utils import load_mnist


def get_adversarial_samples(config, dataset, attack_type):    

    """
    This function will first create a model to attack using attack_model.py then will use the model to create adversarial examples.
    
    args:
    -------------
    
    config : The PredictConfig class instance which already has a session where the model which has model restoration data along with checkpoint location
    
    dataset : The dataset which should be used to create adversarial examples. Ideally a test_data set
    
    supportedAlgorithms : enum object which indicates which algorithm to use to create adversarial examples

    Returns:
    --------------

    Zipped tuples whose first element if the adversarial image and the second image is the 
    """
    #TODO: Extend this so that adversarial examples are created not only for given dataset but also for provided images

    # Restore the saved model and get all paramerters necessary to create adversarial examples
    attack_params = RestoreModel(config)
    #use the attack_params to create a classifier
    classifier = create_classifier(attack_params)
    if(attack_type == SupportedAlgorithms.FSGM):
        x_adv = CreateFGSMAttack(classifier, dataset)
        return x_adv
    elif(attack_type == SupportedAlgorithms.JSMA):
        x_adv = CreateJSMAttack(classifier, dataset)
        return x_adv
    elif(attack_type == SupportedAlgorithms.UAP):
        x_adv = CreateUAPAttack(classifier, dataset)
        return x_adv
    elif(attack_type == SupportedAlgorithms.BA):
        x_adv = CreateSimBattack(classifier, dataset)
        return x_adv
    else:
        return None


def CreateFGSMAttack(classifier, data):  

    '''
    Craft FGSM attack. 
    Args:
    ---------   
    
    data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
    '''  
    adv_crafter = FastGradientMethod(classifier, eps=0.07)
    x_test_adv = adv_crafter.generate(x=data)
    return x_test_adv   

def CreateJSMAttack(classifier, data):    
    _theta = 0.5
    _gamma = 0.05
    adv_crafter = SaliencyMapMethod(classifier=classifier, theta=_theta, gamma=_gamma)
    logger.info("Theta: {}, Gamma: {}".format(_theta, _gamma))
    x_test_adv = adv_crafter.generate(x=data)
    return x_test_adv 

def CreateUAPAttack(classifier, data): 
    attacker_p = {"eps": 0.01}    
    xi = 0.1   
    _max_iter = 30
    logger.info("eps: {}; xi: {}, max_iter: {}".format(attacker_p, xi, _max_iter))
    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #handler = logging.StreamHandler()
    #formatter = logging.Formatter("[%(levelname)s] %(message)s")
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
    adv_crafter = UniversalPerturbation(classifier, attacker="fgsm", attacker_params= attacker_p, eps=xi,max_iter=_max_iter)
    x_test_adv = adv_crafter.generate(x=data)
    noise = adv_crafter.noise
    print(noise.shape)
    logger.info("Fooling rate: {}".format(adv_crafter.fooling_rate))
    for uap in noise:
        visualize_data.plot_image(uap)    
    return x_test_adv 

def CreateSimBattack(classifier, data):    
    #(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    #print(x_train.shape)
    #print(data.shape)
    #Simba needs channel information which is not provided in mnist trainining model

    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #handler = logging.StreamHandler()
    #formatter = logging.Formatter("[%(levelname)s] %(message)s")
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
    _max_itter = 100
    logger.info("BA: Number of iterations: {}".format(_max_itter))
    adv_crafter =BoundaryAttack(estimator=classifier, max_iter=_max_itter, targeted=False)
    x_test_adv = adv_crafter.generate(x=data)
    return x_test_adv 

def create_benign_adv_map(x_test_adv, x_test):
    """
        Create a numpy array whose each element if a tuple containing original image and its adversarial counterpart

        Args:
        ------------------------
        x_test: original/ benign image
        x_test_adv: corresponding adversarial image

    """
    print("shape of adversarial image created {}, shape of the dataset {}:".format(x_test_adv.shape, x_test.shape))
    mapped = zip(x_test, x_test_adv)
    return mapped