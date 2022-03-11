import ModelRepository as mr

"""
    This module should be updated when a new model is added.
"""


def get_model(model_name, precision):
    """
    Returns the model definition of the selected model.

    Args:
    -------

    model_name: SupportedModels enum signifying the name of the model, as specified in the SupportedModels enum

    precision: Precision of the model. This is required because to get the quantized version of the model definition,
                we need to set the bit width in the graph

    Returns:
    ---------

    A modelDesc class implementation that contains 
    
        a. input/output placeholders

        b. Optimizer to be used during training

        c. Model description/ graph

    """
    #FP LeNet models
    if(model_name == mr.SupportedModels.lenet5):
        print("Full precision LeNet5 base Model.")
        return mr.lenet_fp.Model()
    elif(model_name == mr.SupportedModels.lenet5b):
        print("Full precision LeNet5 variant b Model.")
        return mr.lenet_fp_b.Model()
    elif(model_name == mr.SupportedModels.lenet5c):
        print("Full precision LeNet5 variant c Model.")
        return mr.lenet_fp_c.Model()
    elif(model_name == mr.SupportedModels.lenet5tf):
        print("Full precision LeNet5 Model with tf layers.")
        return mr.lenet_tf.Model()
    #FP Models A, B and C
    elif(model_name == mr.SupportedModels.model_a):
        print("Full precision Model A selected.")
        mr.model_a.model_capacity = 16 #32
        return mr.model_a.Model()
    elif(model_name == mr.SupportedModels.model_b):
        print("Full precision Model B selected.")
        mr.model_b.model_capacity = 32 #64
        return mr.model_b.Model()
    elif(model_name == mr.SupportedModels.model_c):
        print("Full precision Model C selected.")
        mr.model_c.model_capacity = 64 #48 #96
        return mr.model_c.Model()

    #Quantized LeNet
    elif(model_name == mr.SupportedModels.lenet_q):
        mr.lenet_q.BITW, mr.lenet_q.BITA, mr.lenet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5".format(mr.lenet_q.BITW, mr.lenet_q.BITA, mr.lenet_q.BITG))
        return mr.lenet_q.Model()
    elif(model_name == mr.SupportedModels.lenet_qb):
        mr.lenet_q_b.BITW, mr.lenet_q_b.BITA, mr.lenet_q_b.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5 one additional FC".format(mr.lenet_q_b.BITW, mr.lenet_q_b.BITA, mr.lenet_q_b.BITG))
        return mr.lenet_q_b.Model()
    elif(model_name == mr.SupportedModels.lenet_qc):
        mr.lenet_q_c.BITW, mr.lenet_q_c.BITA, mr.lenet_q_c.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5 two additional FC".format(mr.lenet_q_c.BITW, mr.lenet_q_c.BITA, mr.lenet_q_c.BITG))
        return mr.lenet_q_c.Model()
        
    #Quantized Models A, B and C
    elif(model_name == mr.SupportedModels.model_aq):
        mr.model_a_q.BITW, mr.model_a_q.BITA, mr.model_a_q.BITG = map(int, precision.split(','))
        mr.model_a_q.model_capacity = 16 #32
        print("Quantization with: {}, {}, {} for MODEL A".format(mr.model_a_q.BITW, mr.model_a_q.BITA, mr.model_a_q.BITG))
        return mr.model_a_q.Model()
    elif(model_name == mr.SupportedModels.model_bq):
        mr.model_b_q.BITW, mr.model_b_q.BITA, mr.model_b_q.BITG = map(int, precision.split(','))
        mr.model_b_q.model_capacity = 32 #64
        print("Quantization with: {}, {}, {} for MODEL B".format(mr.model_b_q.BITW, mr.model_b_q.BITA, mr.model_b_q.BITG))
        return mr.model_b_q.Model()
    elif(model_name == mr.SupportedModels.model_cq):
        mr.model_c_q.BITW, mr.model_c_q.BITA, mr.model_c_q.BITG = map(int, precision.split(','))
        mr.model_c_q.model_capacity = 64 #48 #96
        print("Quantization with: {}, {}, {} for MODEL C".format(mr.model_c_q.BITW, mr.model_c_q.BITA, mr.model_c_q.BITG))
        return mr.model_c_q.Model()
    else:
        print("nothing to return")

def get_mod_typ(training_model, precision):

    """
        This function returns SupportedModels Enum based on the string input. The string input is usually from a yaml file.

        Args:
        -----------------
        training_model: name of the model as specified in the user input from yaml file (string)

        precision: precision of the model (string of format 2,2,32 representing bitW, bitA and bitG). Is None for full precision


        Returns:
        ------------------
        A "SupportedModels" type enum
    
    """

    #Get the model architecture
    if(training_model.lower() == "lenet5" or training_model.lower() == "lenet5_a"):
       model_arch = mr.SupportedModels.lenet5
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.lenet_q
    elif(training_model.lower() == "lenet5_b"):
       model_arch = mr.SupportedModels.lenet5b
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.lenet_qb
    elif(training_model.lower() == "lenet5_c"):
       model_arch = mr.SupportedModels.lenet5c
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.lenet_qc
    elif (training_model.lower() == "lenet5tf"):
        model_arch = mr.SupportedModels.lenet5tf
        if(precision is not None):
           model_arch = mr.SupportedModels.lenet_q
    #MODEL A, B , C MNIST
    elif (training_model.lower() == "model_a"):
        model_arch = mr.SupportedModels.model_a
        if(precision is not None):
           model_arch = mr.SupportedModels.model_aq
    elif (training_model.lower() == "model_b"):
        model_arch = mr.SupportedModels.model_b
        if(precision is not None):
           model_arch = mr.SupportedModels.model_bq
    elif (training_model.lower() == "model_c"):
        model_arch = mr.SupportedModels.model_c
        if(precision is not None):
           model_arch = mr.SupportedModels.model_cq    
    else:
        print("unimplemented")
        return
    return model_arch



def get_mod_typ_cifar(training_model, precision):

    """
        This function returns SupportedModels Enum based on the string input. 
        The string input is usually from a yaml file.
        This function is specifically for CIFAR10 models only since the MNIST models were too many and made the code conjusted

        Args:
        -----------------
        training_model: name of the model as specified in the user input from yaml file (string)

        precision: precision of the model (string of format 2,2,32 representing bitW, bitA and bitG). Is None for full precision


        Returns:
        ------------------
        A "SupportedModels" type enum
    
    """
    #Get the model architecture
    if(training_model.lower() == "cifar_a" or training_model.lower() == "cifara"):
       model_arch = mr.SupportedModels.cifar_a
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.cifar_aq
    elif(training_model.lower() == "cifar_b" or training_model.lower() == "cifarb"):
       model_arch = mr.SupportedModels.cifar_b
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.cifar_bq
    elif(training_model.lower() == "cifar_c" or training_model.lower() == "cifarc"):
       model_arch = mr.SupportedModels.cifar_c
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.cifar_cq

    #Get the resnet architecture
    elif(training_model.lower() == "resnet3" or training_model.lower() == "resnet_3"):
       model_arch = mr.SupportedModels.resnet_3
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.resnet_3q
    elif(training_model.lower() == "resnet5" or training_model.lower() == "resnet_5"):
       model_arch = mr.SupportedModels.resnet_5
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.resnet_5q
    elif(training_model.lower() == "resnet7" or training_model.lower() == "resnet_7"):
       model_arch = mr.SupportedModels.resnet_7
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = mr.SupportedModels.resnet_7q
    else:
        print("unimplemented")
        return
    return model_arch

def get_model_cifar(model_name, precision):
    """
    Returns the model definition of the selected model. Again, specifically for cifar

    Args:
    -------

    model_name: SupportedModels enum signifying the name of the model, as specified in the SupportedModels enum

    precision: Precision of the model. This is required because to get the quantized version of the model definition,
                we need to set the bit width in the graph

    Returns:
    ---------

    A modelDesc class implementation that contains 
    
        a. input/output placeholders

        b. Optimizer to be used during training

        c. Model description/ graph

    """
    #FP CIFAR models
    if(model_name == mr.SupportedModels.cifar_a):
        print("Full precision CIFAR-10 covnet base Model A.")
        return mr.cifar_convnet.Model()
    if(model_name == mr.SupportedModels.cifar_b):
        print("Full precision CIFAR-10 covnet base Model B.")
        return mr.cifar_convnet_b.Model()
    if(model_name == mr.SupportedModels.cifar_c):
        print("Full precision CIFAR-10 covnet base Model C.")
        return mr.cifar_convnet_c.Model()

    #FP CIFAR Resnet models
    if(model_name == mr.SupportedModels.resnet_3):
        print("Full precision ResNet model with n = 3 and 20 layers")
        return mr.cifar10_resnet.Model(3)
    if(model_name == mr.SupportedModels.resnet_5):
        print("Full precision ResNet model with n = 5 and 32 layers")
        return mr.cifar10_resnet.Model(5)
    if(model_name == mr.SupportedModels.resnet_7):
        print("Full precision ResNet model with n = 7 and 44 layers.")
        return mr.cifar10_resnet.Model(7)


        
    #Quantized versions of CIFAR10 models
    elif(model_name == mr.SupportedModels.cifar_aq):
        mr.cifar_convnet_q.BITW, mr.cifar_convnet_q.BITA, mr.cifar_convnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model A.".format(mr.cifar_convnet_q.BITW, mr.cifar_convnet_q.BITA, mr.cifar_convnet_q.BITG))
        return mr.cifar_convnet_q.Model()
    elif(model_name == mr.SupportedModels.cifar_bq):
        mr.cifar_convnet_qb.BITW, mr.cifar_convnet_qb.BITA, mr.cifar_convnet_qb.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model B.".format(mr.cifar_convnet_qb.BITW, mr.cifar_convnet_qb.BITA , mr.cifar_convnet_qb.BITG))
        return mr.cifar_convnet_qb.Model()
    elif(model_name == mr.SupportedModels.cifar_cq):
        mr.cifar_convnet_qc.BITW, mr.cifar_convnet_qc.BITA, mr.cifar_convnet_qc.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model C.".format(mr.cifar_convnet_qc.BITW, mr.cifar_convnet_qc.BITA , mr.cifar_convnet_qc.BITG))
        return mr.cifar_convnet_qc.Model()

    #Quantized versions of ResNet models
    elif(model_name == mr.SupportedModels.resnet_3q):
        mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA, mr.cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=3.".format(mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA, mr.cifar10_resnet_q.BITG))
        return mr.cifar10_resnet_q.Model(3)
    elif(model_name == mr.SupportedModels.resnet_5q):
        mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA, mr.cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=5.".format(mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA , mr.cifar10_resnet_q.BITG))
        return mr.cifar10_resnet_q.Model(5)
    elif(model_name == mr.SupportedModels.resnet_7q):
        mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA, mr.cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=7.".format(mr.cifar10_resnet_q.BITW, mr.cifar10_resnet_q.BITA , mr.cifar10_resnet_q.BITG))
        return mr.cifar10_resnet_q.Model(7)
    
    else:
        print("nothing to return")