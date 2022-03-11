import os

from tensorpack import *
from tensorpack.callbacks import graph
from tensorpack.dataflow import dataset
#from tensorpack.predict.base import CreateAdvAttacks
from tensorpack.tfutils.sessinit import SmartInit
from enum import Enum

from tensorpack.input_source import PlaceholderInput
from tensorpack.tfutils.common import get_tensors_by_names, get_op_tensor_name
from tensorpack.tfutils.tower import PredictTowerContext
#if i call this, this will call the updated version of the file in my PC but without it tensorpack import * will call the defaule version somehow
#from tensorpack.predict.base import OfflinePredictor

#Get custom Mnist data extraction (default tensorflow has an addition dimention that can cause problems during inference)
from DataSets.mnist import GetMnist
from Inference.inference_core import infer_model, show_one_predictions, infer_model_dataset, multiple_infer_model_dataset
import save_restore_images
import visualize_data
import ModelRepository as mr
from DataSets.cifar import Cifar10, get_cifar10_data, getaugmenteddata_with_all_images


#save location for anything that needs to be saved apart from training related data
save_location = os.path.join("results")
#Mnist data location
Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"
Cifar_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\CIFAR10"

def get_prediction_config(saved_model, _model):
    """ Prediction config sets up all necessary configuration necessary for inference.

     It sets up:

           1. Input signatures   

           2. Session

           3. Tower function using the model() and input signatures 

           4. Output names 
           
     These are used later by offline predictor to setup inference graph """

    session = SmartInit(saved_model)

    predictorConfig = PredictConfig(
        model= _model,
        session_init= session,
        input_names=['input'],
        output_names=['output'])
    return predictorConfig


def get_dataset(inference_image_source):
    """
        Get the dataset iterable from the user speficied dataset

        Args:
        ------------
        inference_image_source: dataset name. for eg: mnist OR it can also be a npz file containing image, labels and index numpy arrays
   
    """
    if(inference_image_source.lower() == "mnist"):
        #get dataset to perform inference with
        data_test = GetMnist('test', shuffle= False , dir=Mnist_Data_Dir)  
        #x_test = data_test.images
        #labels = data_test.labels
    elif(inference_image_source.lower() == "cifar10"):
        data_test = get_cifar10_data('test', dir= Cifar_Data_Dir)
    else:
        print("Restoring images from location: {}".format(inference_image_source))
                
        #data_test = save_restore_images.save_or_load_image__npz("load", inference_image_source)
        """
            data_test here again is an iterable object with each iteration returning the image and corresponding label.
            if the loaded npz file contains adversarial image data, this will load that as well in a adversarial_image field.
            The iterable data_test in this case will then also return "adversarial_images" as a iterable component after benign_images and labels.
        """
        data_test = save_restore_images.save_or_load_image__npz("load", inference_image_source)
    return data_test


def inference_mode(predictorConfig, _dataset, adversarial_inference=False):  
    """
        Runs inference on the provided image data. Computes accuracy and adversarial accuracy (whenever possible)

        Args:
        --------
        predictorConfig: PredictConfig class that takes current session where the graph is restored and also contains the 
                            Model Description
        

        dataset: an iterable object that has image and label as a yield
        
    """

    #data_test = GetMnist('test', Mnist_Data_Dir) 
    #data_test_adv = save_restore_images.save_or_load_image__npz("load", 
    #r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\trained_images\MNIST-2,2,32\mnist_conv_adv_pre-2,2,32.npz") 
    #x_test = data_test.images[data_test_adv.image_index]
    #labels = data_test.labels[data_test_adv.image_index]


    #data_test = get_cifar10_data('test', Cifar_Data_Dir)     
    #data_test = save_restore_images.save_or_load_image__npz("load", 
    #r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\logs\trained_images\cifar10-FP\correct_pred_images.npz") 
    #data_test = getaugmenteddata_with_all_images(data_test)
    #x_test = data_test.images
    #labels = data_test.labels
    #index =  data_test.image_index

    #data labels, one of these will have high prob than others during inference
    output_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    output_labels_cifar = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    #if the image contains adversarial image data, then get the adversarial image also
    #get_adv_accuracy = False

    #im = Image.fromarray(x_test[2])
    #im.save(r'C:\tmp\SaveImages\{}.png'.format(y_test[2]))
    #predictor = OfflinePredictor(predictorConfig)
    #show_one_predictions(data_test.images[502], data_test.labels[502], predictor, output_labels_cifar)
    #infer_model_dataset(predictorConfig, dataset=x_test, labels= labels, index= index)
    accuracy_t1, error_t1, saved_corr_images, saved_false_images, agvl2, avglinf = infer_model(
        predictorConfig, 
        output_labels, 
        dataset=_dataset, 
        get_adv_accuracy=adversarial_inference)  
        
    return accuracy_t1, error_t1, saved_corr_images, saved_false_images, agvl2, avglinf
    
def initiate_inference(userconfig):
    """
        Take the user configuration from the yaml file and initiate inference on the specified model.

        To initiate inference this function will:

                    a.  Set up a predictor config that uses the current session to establish saved variables from checkpoint

                    b.  Set up model graph via the get_model method

                    c.  Perform inference using the setup model graph, variables and return accuracy (adversarial or normal)


        Args:
        -------------
        userconfig: A dictionary object retured from yaml data loader. This object should contain details like:

                    a.  What model should be used to perfrom inference (saved model)

                    b.  What graph to use (modelDesc)

                    c.  Is it quantized?

                    d.  If it is a quantized model, then what bitwidth to use?

                    e.  What dataset to use? When dataset is given it will perform inference in dataset otherwise on imageset

                 


    TODO: This function does a bit "too much". Need to find a better way. May be skip the whole configuration part and 
            start training directly...

    """  
    #get the saved model from disk
    inference_saved_model = userconfig["inference-options"]["load-model"]
    
    #the precision here should be a comma separated value viz: 2,2,32
    precision = userconfig["inference-options"]["precision"]["bitwidth"]  
    
    #get the image source. The image source could be entire dataset like mnist or it can be specific set of images in npz format  
    inference_image_source = userconfig["inference-options"]["images"]
    #get the base model description. The base model description should match the saved model description
    inference_model_description = userconfig["inference-options"]["base-model"]  
    
    #setup logger
    dataset_name_log = inference_image_source
    if(precision is None):
        log_name = "FP"
    else:
        log_name = precision    
    logger.set_logger_dir(os.path.join('logs','inference_log', '{}-{}'.format(dataset_name_log, log_name)))
    
    #get model architecture
    #model = mr.get_model(mr.get_mod_typ(inference_model_description, precision), precision)
    
    #if the dataset is mnist get mnist models, if the dataset is cifar get cifar models
    if(inference_image_source == "mnist"):
        model = mr.get_model(mr.get_mod_typ(inference_model_description, precision), precision)
    elif(inference_image_source == "cifar10"):
        model = mr.get_model_cifar(mr.get_mod_typ_cifar(inference_model_description, precision), precision)
    
    #get data set to run the inference. When a set of images in given in a npz file this method will create a iterable 
    #object that we can use as a normal dataset (iterable object)
    data_test = get_dataset(inference_image_source)
    
    #setup configurations for restoring model for predictions and for creating adversarial attacks
    predictorConfig = get_prediction_config(inference_saved_model, model) 

    #perform inference by running network on the inference mode
    inference_mode(predictorConfig, data_test)