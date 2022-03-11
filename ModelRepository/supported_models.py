from enum import Enum
'''
    This class lists all the supported models.
    
    A _q suffix indicates the model is quantized version while tf in the end means that the model uses "tensorflow layers"
    in leu of symbolic input supported by tensorpack
'''

#List all supported models
class SupportedModels(Enum):
    #MNIST MODELS
    #standard LeNet models. The app was thoroughly tested with these models before other conv models were added.
    lenet5tf = "lenet5tf"
    lenet5 = "lenet5" 
    lenet5b = "lenet5b" 
    lenet5c = "lenet5c"

    lenet_q = "lenet_q"
    lenet_qb = "lenet_qb"
    lenet_qc = "lenet_qc"

    #Still MNIST models but using different architecture (more parameters)
    model_a = "model_a"
    model_aq = "model_aq"    #quantized version of model A
    model_b = "model_b"
    model_bq = "model_bq"    #quantized version of model B
    model_c = "model_c"
    model_cq = "model_cq"    #quantized version of model C


    #CIFAR MODELS
    #CNN based on CIFAR datasets
    cifar_a = "cifar_a"
    cifar_b = "cifar_b"
    cifar_c = "cifar_c"
    #quantized version of CIFAR datasets
    cifar_aq = "cifar_aq"
    cifar_bq = "cifar_bq"
    cifar_cq = "cifar_cq"

    #RESNETS TYPE CNNs CIFAR datasets
    resnet_3 = "resnet_3"
    resnet_5 = "resnet_5"
    resnet_7 = "resnet_7"
    #quantized version of ESNETS TYPE CNNs
    resnet_3q = "resnet_3q"
    resnet_5q = "resnet_5q"
    resnet_7q = "resnet_7q"

