import os


from tensorpack import *
#if i call this, this will call the updated version of the file in my PC but without it tensorpack import * will call the defaule version somehow
#from tensorpack.predict.base import OfflinePredictor

#Get custom Mnist data extraction (default tensorflow has an addition dimention that can cause problems during inference)
from DataSets.mnist import GetMnist
from DataSets.cifar import Cifar10, get_cifar10_data

import ModelRepository as mr

#save location for anything that needs to be saved apart from training related data
save_location = os.path.join("results")
#Mnist data location
Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"
Cifar_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\CIFAR10"


def get_config_mnist(_model, data_train, data_test):
    """
    Get training configuration for MNIST. The training configuraion basically includes:

        1. How many checkpoints to save 

        2. What to include in call backs (validation accuracy, loss and so on)

        3. Number of epochs to train

        4. The model to use
     
    AutoResumeTrainConfig() is also used by Tensorpack to enable resuming training from previously saved checkpoint.
    For more information pls follow the guide on Tensorpack: https://tensorpack.readthedocs.io/en/latest/modules/train.html#tensorpack.train.AutoResumeTrainConfig
    
    """
    return AutoResumeTrainConfig(
        data=QueueInput(data_train), 
        #as defined in input_source.inputsource.queueinput.
        #Datatrain object, while iterating, removes last dimension on MNIST is removed, presumably because it is 1 channel (28,28,1) can be same as (28,28)
        #so for now just know that the input signature must be 28x28 and later add dimension
        callbacks=[
            ModelSaver(max_to_keep=20),   #save model after each epoch
            
                # Run inference(for validation) after every epoch
                # data_test is not ideal if we have validation split but now data is not split so use test data. This is a mal-practice. Will correct this later
                
                # scalarStats gives the average of scalar value specified over all datapoints. ScalarStats('error_rate') is different however, it 
                # seems that it also considers batchsize during taking average i.e average of error over batches. But there is another keyword
                # classificationerror which  is different from scalarstats("error_rate")
                # Classification error also considers that batch size might be different while computing average (classification error is 
                # basically the exact opposite of accuracy (classificationaccuracy)). It also considers that value computed over averaged batches is not same as overall average.
                # BUT BOTH CLASSIFICATION ERROR AND CLASSIFICAION ACCURACY STILL GIVE DIFFERENT ACC THAN MINE WHEN BATCH SIZE=128,
                # THIS MEANS THAT IT DOES NOT CONSIDER THAT DATA MIGHT NOT BE EXACTLY DIVIDED BY THE BATCH SIZE, 
                # ClassificationError gives "true" error this is different that ScalarStats("error-rate") because here ScalarStats does not consider batch size 
                # difference and just give average over the data points.
                # However, this is true for training (scalar stats is only for inference, so inference is also done in batches), for testing it seems that batch-size is 1 because inference accuracy over all batches is equal
                # to how I was computing accuracy (individual data points), BUT THE ACCURACY COMES DIFF WHEN TOTAL DATA SIZE IS NOT EXACTLY 
                # DIVISIBLE BY BATCH SIZE (600000/128) IN THIS CASE MY ACCURACY DIFFERS FROM COMPUTED ACCURACY 
                # SO TAKE ALWAYS BATCH SIZE THAT CAN  BE DIVIDED EXACTLY BY TOTAL DATA IF YOU WANT TO VERIFY ACCURACY OTHERWISE IT IS ALSO OK TO TAKE BATCH SIZE 128
                # WHEN BATCH SIZE DOES NOT EXACTLY DIVIDE THE TRANING DATA THEN THE RESULTING ACCURACY AVERAGED OVER BATCHES (PER BATCH AVERAGE ACC. THEN AGAIN AVERAGED BY NUMBER OF BATCHES IN THE END)
                # is not so different from what is computed manually i.e the former has more precision (like 0.83334) than computed by me (one per batch) is simply 0.84
                # More at:
                    #https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScalarStats
            
            #using "total_Cost" or "cross_enropy" is same in case of inference since in inference we do not consider regulaization cost
            
            #tested: ClassificationError() gives true error irrespective of the batches. Tested in Resnet when the batchsize for inference was 64 
            #accuracy was different than mine in inference mode but the classification error was same

            InferenceRunner(data_test,  
                            [ScalarStats(['accuracy', "cross_entropy_loss"], prefix="val")]) 
            #, DumpTensors(["b_quant:0", "a_quant:0"]) # for weights 
            #,DumpTensors(["IdentityN_12:0"])            
            ,MaxSaver('val_accuracy')
            ,MinSaver("val_cross_entropy_loss")
        ],        
        model= _model,
        max_epoch=1000,
    )


def get_config_cifar(_model, data_train, data_test):

    """
    Training configuration is as per the example provided in Tensorpack repo example for training CIFAR data (not specifically cifar10).
    The model is available at: https://github.com/tensorpack/tensorpack/blob/master/examples/basics/cifar-convnet.py

    Get training configuration for CIFAR-10. The training configuraion basically includes:

        1. How many checkpoints to save 

        2. What to include in call backs (validation accuracy, loss and so on)

        3. Number of epochs to train

        4. The model to use

        5. In addition, when training conv model in CIFAR we also have to vary the learning rate. This variation is done based on
            given scalar parameter (val. loss in our case, original authors use validation accuracy but we found that val. loss provides
            better models in case of quantized versions and also in case of full precision, the model is slightly better)

            Not using this variation in lr the model is very hard to converge and will not give desired results.    
    """

    def lr_func(lr):
        if lr < 3e-5:
            raise StopTraining()
        return lr * 0.31
    return AutoResumeTrainConfig(
        data=QueueInput(data_train), 
        callbacks=[
            ModelSaver(max_to_keep=10),   
            InferenceRunner(data_test,  
                            [ScalarStats(['accuracy', "cross_entropy_loss"])])       
            ,MaxSaver('validation_accuracy')
            ,MinSaver("validation_cross_entropy_loss")
            # details about the statMonitorParamSetter is at: 
            # https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.StatMonitorParamSetter
            # so basically if the selected scalar param does not decrease (increase if reverse=true is used) by threshold then the 
            # defined function is called, in this case the learning rate is decreased.
            # see callbacks.TensorPrinter(names)
            #,StatMonitorParamSetter('learning_rate', 'validation_accuracy', lr_func, threshold=0.001, last_k=10, reverse=True)
            #for quanitzed nets below is better:
            ,StatMonitorParamSetter('learning_rate', 'validation_cross_entropy_loss', lr_func, threshold=0.001, last_k=10)
        ],        
        model= _model,
        max_epoch=400,
    )


def get_config_cifar_resnet(_model, data_train, data_test):
    """

    Training configuration is as per the example provided in Tensorpack repo example for training ResNet on CIFAR data (not specifically cifar10).
    The model is available at: https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py
    
    Get training configuration for CIFAR-10. The training configuraion basically includes:

        1. How many checkpoints to save 

        2. What to include in call backs (validation accuracy, loss and so on)

        3. Number of epochs to train

        4. The model to use

        5. In addition, when training Resnets in CIFAR learning rates are varied in the given intervals. This is not experimented my me
            but used the defaults from authors which seemed to give proper results at least for experiments.
            
    """

    #run two parallel processes. ONLY FOR TRAINING DO NOT USE FOR TESTING.
    #relevant documentation at: https://tensorpack.readthedocs.io/en/latest/modules/dataflow.html#tensorpack.dataflow.MultiProcessRunner
    data_train = MultiProcessRunner(data_train, 3, 2)
    return TrainConfig(
        data=QueueInput(data_train), 
        callbacks=[
            ModelSaver(max_to_keep=10),   
            InferenceRunner(data_test,  
                            [ScalarStats(['accuracy', "cross_entropy_loss"])])       
            ,MaxSaver('validation_accuracy')
            ,MinSaver("validation_cross_entropy_loss"),
            ScheduledHyperParamSetter('learning_rate',
            [(1, 0.1), (32, 0.01), (48, 0.001), (72, 0.0002), (82, 0.00002)])
        ],        

        #documentation for steps_per_epochs at https://tensorpack.readthedocs.io/en/latest/modules/train.html

        #basically the steps_per_epochs do not affect the sequence of data seen by the model. So model is trained normally.
        #but an epoch now is made longer to adjust the call backs. So this value only affects callbacks. So since batch size is 
        #64 the total data seen by model in 1 traditional epoch will be 50000/64 but after this normally epoch ends and callbacks
        #take place but in this case the training continues until the total steps have reached 1000 so an epoch is longer.
        #An epoch here means how often callbacks are scheduled and not how much times the model is seeing the same data.
        steps_per_epoch=1000,
        model= _model,
        max_epoch=64,
    )

def get_dataset(dataset_name):

    """
        Get the dataset iterable object that will iterate over the specified dataset. 
        Each iteration should yield Image and Label

        Args:
        ------------------
        Name of the dataset (String) as read from the yml file.

        Returns:
        -------------------

        A test data and train data (iterable) objects 
    """

    if(dataset_name.lower() == "mnist"):        
        # prepare dataset
        # get data from tar. The Mnist is not downloaded each time but make sure it is in folder: 
        # C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data
        
        # GetMnist returns (data_train/ data_test) an iterable object which returns a list [image and data] in each iteration (as defined by __iter__). Apart from this we can also get
        # individual images/labels via the _cache dictionary within data_train.
        data_train = GetMnist('train', shuffle=True, dir=Mnist_Data_Dir) # has dictionary data_test._cache["train/test/extra"] that holds data   
        data_test = GetMnist('test', shuffle=False, dir=Mnist_Data_Dir) 
        
        # No need to resize the images
        # resize image to 32x32
        # augmentors = [
        #        imgaug.Resize((32, 32)),
        #]
        # apply image augmentators
        # data_train is not an array but an object
        # data_train = AugmentImageComponent(data_train, augmentors)  
        """
        Separate data in batches, consider remainder as 60000/128 is 468.75 so total batched required will be 469
            Batched data() takes batch number and current total dataset (60000, 28, 28) and  returns an iterable list
            whose first element is the batched image data and second element is correct labels,
            so,
            list[0].shape = [128,28,28] list[1]=[128] so we see data is batched by defined stacks and then returned in batches along with
            relevant labels. This batched data iterable is then fed as input to "QueueInput" which then will queue the data in batches"
            ref:
                1. https://tensorpack.readthedocs.io/en/latest/tutorial/trainer.html#tower-trainer
                2. https://tensorpack.readthedocs.io/en/latest/tutorial/training-interface.html#with-modeldesc-and-trainconfig
        """   
        data_train = BatchData(data_train, 128, remainder=True) #as defined in dataflow.common.batchdata
        #data_train = MultiProcessRunner(data_train, 5,3) 
        data_test = BatchData(data_test, 100, remainder=True) #as defined in dataflow.common.batchdata
        #for i in data_train:
        #    print("batched data ", i[0].shape)
        # This does not work on windows, may be MultiProcessRunner(data_train, 5) works but not tested    
        #data_train = MultiProcessRunnerZMQ(data_train, 5)

    elif (dataset_name.lower() == "cifar10"):        
        data_train = get_cifar10_data('train', dir=Cifar_Data_Dir) # has dictionary data_test._cache["train/test/extra"] that holds data   

        data_test = get_cifar10_data('test', dir=Cifar_Data_Dir) 


        #Since the batch numbers and other details might be changing due to change in dataset the data_test and data_train are kept redundant with MNIST
        #data_train = BatchData(data_train, 128, remainder=True) #as defined in dataflow.common.batchdata
        data_train = BatchData(data_train, 64, remainder=True) #take batch-size as 64 when training resnet
 
        data_test = BatchData(data_test, 100, remainder=True) #as defined in dataflow.common.batchdata
        #data_test = BatchData(data_test, 64, remainder=True) #as defined in dataflow.common.batchdata

    else:
        print("datasets other than MNIST and CIFAR10 is not currently implemented")
        return

    return data_train, data_test


def initiate_traning(userconfig):
    """
        Take the user configuration from the yaml file and initiate training of the specified model.

        To initiate training this function will:

                    a.  Get dataset iterable which will be used by tensorpack during training.

                    b.  Get the model definition

                    c.  Set configuration for traning

                    d.  Launch training with the specified configuration 

        Args:
        -------------
        userconfig: A dictionary object retured from yaml data loader. This object should contain details like:

                    a.  What model to train

                    b.  Is it quantized?

                    c.  What dataset to use?

                    d.  If it is a quantized model, then what bitwidth to use?


    TODO: This function does a bit "too much". Need to find a better way. May be skip the whole configuration part and 
            start training directly...

    """  
    #get the name of the model to train. 
    training_model = userconfig["training-options"]["model"]
    
    #the precision here should be a comma separated value viz: 2,2,32
    precision = userconfig["training-options"]["precision"]["bitwidth"]
    
    #get the name of the dataset. MNIST/CIFAR
    dataset_name = userconfig["training-options"]["dataset"]
    
    #setup logger  
    if(precision is None):
        log_name = "FP"
    else:
        log_name = precision.replace(',', '')

    logger.set_logger_dir(os.path.join('logs','train_log', '{}-{}'.format(dataset_name, log_name)))    
    
    #get model architecture
    if(dataset_name == "mnist"):
        model = mr.get_model(mr.get_mod_typ(training_model, precision), precision)
    elif (dataset_name == "cifar10"):
        model = mr.get_model_cifar(mr.get_mod_typ_cifar(training_model, precision), precision)  
    else:
        print("Could not find the selected mode.") 
        return
        
    #get data set
    data_train, data_test = get_dataset(dataset_name)
    print(dataset_name)
    #setup configuration for traning
    is_resnet_model = False #some changes in the training configuration when resnet model is being trained so this bool is necessary
    if(dataset_name == "mnist"):
        print("mnist selected")
        config = get_config_mnist(model, data_train, data_test)  
    elif(dataset_name == "cifar10"):
        print("cifar selected")
        config = get_config_cifar(model, data_train, data_test)  
        if(mr.SupportedModels.resnet_3 == mr.get_mod_typ_cifar(training_model, precision) or 
            mr.SupportedModels.resnet_5 ==  mr.get_mod_typ_cifar(training_model, precision) or 
            mr.SupportedModels.resnet_7 == mr.get_mod_typ_cifar(training_model, precision) or
            mr.SupportedModels.resnet_3q == mr.get_mod_typ_cifar(training_model, precision) or
            mr.SupportedModels.resnet_5q == mr.get_mod_typ_cifar(training_model, precision) or
            mr.SupportedModels.resnet_7q == mr.get_mod_typ_cifar(training_model, precision)):
            is_resnet_model = True 
            #the training config for resnet is different  
            config = get_config_cifar_resnet(model, data_train, data_test)                    
        else:
            is_resnet_model = False
        
    print("Is Resnet?", is_resnet_model)  

    #launch training with defined configurations
    launch_train_with_config(config, SimpleTrainer()) 
