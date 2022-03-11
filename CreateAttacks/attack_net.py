import os
import numpy as np

from tensorpack import *
#if i call this, this will call the updated version of the file in my PC but without it tensorpack import * will call the defaule version somehow
#from tensorpack.predict.base import OfflinePredictor

#Get custom Mnist data extraction (default tensorflow has an addition dimention that can cause problems during inference)
from DataSets.mnist import GetMnist
from CreateAttacks.supported_attacks import SupportedAlgorithms
from CreateAttacks.create_adversarial_samples import get_adversarial_samples
import save_restore_images
import visualize_data
import Inference
import ModelRepository as mr
import os
from DataSets.cifar import Cifar10, get_cifar10_data, getaugmenteddata_with_all_images

Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"
Cifar_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\CIFAR10"

Adv_sample_number_Src = 2000            #number of adversarial examples that will be selected from source
# From "Adv_sample_number_Src" select "Adv_sample_number_total" 
Adv_sample_number_total = 1000          #number of adv examples that have their benign counter part classified correcltly by both source and target

def create_attack_mode(config, algorithm,  precision, data_test, source_data):
    """
    From the correctly classified images in source_data create adversarial images.

    Args
    ----

    config : The PredictConfig class instance which already has a session where the model which has model restoration data along with checkpoint location

    algorithm: Adversarial examples generation algorithm

    precision: Precision of the network that will be used when restoring the model graph

    data_test: Set of correctly predicted images from the original dataset

    source_data: Original dataset (MNIST/ CIFAR)

    """
   
    #data labels, one of these will have high prob than others during inference
    output_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
   
    #get the image index from the data that were correctly classified classified by the net to select random index to craft 
    # adversarial examples
    image_index = data_test.image_index
    
    #CREATE ADVERARIAL IMAGES HERE
    #define how many images to produce
    limit = 1500
    #define number of sets of adversarial images to create. Transferability from each of these sets will be then averaged later
    iterations = 3
    for i in range(iterations):
        #randomly select the images from the correctly predicted data based on the correctly predicted data
        random_index = np.random.choice(image_index, Adv_sample_number_Src, False)
        #from the original source data (MNIST SET), when crafting adversarial images, now use only the selected random 
        
        # select images from the source dataset. Since the index are already correctly predicted data we can use source_data(mnist/cifar) here
        craft_examples = source_data.images[random_index]
        
        # craft adversarial attacks
        adv_samples = get_adversarial_samples(config, craft_examples, algorithm)
        #adv_samples = CreateFGSMAttack(attack_params, x_test) 
        #Save adversarial examples to numpy array    
        #saved_images = save_restore_images.save_or_load_image__npz("save", os.path.join("TrainedImages", "mnist_conv_full_pre.npz"), adv_samples[:limit], labels[:limit])
        
        savedir = logger.get_logger_dir()
        #image_index = get_image_index(adv_samples)
        
        saved_images = save_restore_images.save_or_load_image__npz(
            "save", 
            os.path.join(savedir, "mnist_conv_adv_pre-{}--run-{}.npz".format(precision, i)), 
            #benign_image= x_test[:Adv_sample_number_Src], 
            image= adv_samples, 
            labels=source_data.labels[random_index], 
            image_index=random_index)
        #save first 10 images to disc
        #save_restore_images.save_image(adv_samples[:100], r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\Results", False, labels)
        visualize_data.plot_images(adv_samples)
        #infer_model(config, adv_samples)
        print("Adversarial images were crafted using {} algorithm and were saved in location {}".format(algorithm, os.path.join(savedir, "mnist_conv_adv_pre-{}.npz".format(precision))))
        #accuracy_t1, error_t1 = infer_model(config, output_labels, saved_images, purpose="attack")
        #print("accuracy: ", accuracy_t1)
        #print("error: ", error_t1)
 
# Get the image index.
# Since we are creating adversarial examples from sequential images (top 10) so just get sequential array as image index
def get_image_index(data_samples):
    return(np.arange(1, data_samples.shape[0]+1, 1))  

def get_attack_algorithm(algorithm_str):
    """
        Read the string from user and return a enum specifying which algorithm should be used to craft adversarial examples
    
        Returns:
        -----------------

        Enum denoting the attack algorithm that will be used to craft the adversarial images
    
    """
    if(algorithm_str.lower() == "fgsm"):
        supported_alg = SupportedAlgorithms.FSGM
    elif (algorithm_str.lower() == "jsma"):
        supported_alg = SupportedAlgorithms.JSMA
    elif (algorithm_str.lower() == "uap"):
        supported_alg = SupportedAlgorithms.UAP
    elif (algorithm_str.lower() == "ba"):
        supported_alg = SupportedAlgorithms.BA
    else:
        print("The {} algorithm is not supported.".format(algorithm_str))
        return  

    return supported_alg 

def get_data(dataset_name):
    """
    Based on the user input in the yaml file, fetch an iterable object which yields dataset to create attacks on the network
    
    Args:
    ----------
    dataset_name: name (string) of the dataset from which the attacks are to be created.

    Returns:
    ---------
    An iterable object that yields images and lables for attack creation

    """
    if (dataset_name.lower() == "mnist"):
        data_test = GetMnist('test', shuffle=False, dir=Mnist_Data_Dir)
    elif (dataset_name.lower() == "cifar10"):
        data_test = get_cifar10_data('test', dir= Cifar_Data_Dir)
    else:
        print("The mentioned dataset is not implemented")
        return
    return data_test


def get_prefiltered_images(saved_corr_images, adv_images):
    """
    Get a dataset containing images, labels and index (index here means index corresponding to source db (like MNIST)) that are
    correctly classified by both SOURCE and the TARGET db


    The logic here is that, every image in adv_images has been crafted from source net where the corresponding clean image was correctly
    classified by the source network.
    then, consider only adv_images whose clean versions have been classified correctly by both target(img_data) and source

    Args:
    -------
    saved_corr_images: An iterable object of dataset which yields all the images correctly classified by the target (Current net)

    adv_images: A .npz containing images that are adversarial in nature and are created at source net (npz should have image, labels, index)
    

    Returns:
    --------------
    An iterable dataset that contains adversarial images whose benign counter parts where classified correclty by both source and target network
    
    """
    print("Getting images that are correctly classified by both source and target(this) network...")
    
    adv_data = save_restore_images.save_or_load_image__npz("load", adv_images)
    cpi_st = np.empty([0,28,28]) # create a empty numpy array to hold all the correctly predicted images from both source and target
    correct_labels = []         # List containing all the correct labels (later change to numpy while saving to file)
    image_index = []            #image index of all the correctly predicted data
    total = 1000                #how adversarial images to take
   
    for img_data in saved_corr_images: 
        corr_image = (img_data[0])[np.newaxis, :, :]
        corr_label = img_data[1]
        corr_image_index = img_data[2]
        for img_data_adv in adv_data: 
            #compare the image index of both the source and the targe db.
            #please be aware that the "image-index" should always be from a source db like MNIST or CIFAR           
            if(corr_image_index == img_data_adv[2]):
                adv_image = (img_data_adv[0])[np.newaxis, :, :]
                cpi_st = np.append(cpi_st, adv_image, axis=0)
                list.append(correct_labels, corr_label)
                image_index.append(corr_image_index)
                #no need to compare forward when match is found
                break
                #print("match{},{}", corr_image_index, img_data_adv[2])
        if(len(image_index) >= Adv_sample_number_total):
            break
    savedir = logger.get_logger_dir() 
    save_prefiltered_images = save_restore_images.save_or_load_image__npz(
        "save", 
        os.path.join(savedir, "mnist_prefiltered_images.npz"),
        image= cpi_st, 
        labels=np.array(correct_labels), 
        image_index= np.array(image_index)) 
     
    return save_prefiltered_images


def get_prefiltered_images_fast(saved_corr_images, adv_images):
    """
    Get a dataset containing images, labels and index (index here means index corresponding to source db (like MNIST)) that are
    correctly classified by both SOURCE and the TARGET db


    The logic here is that, every image in adv_images has been crafted from source net where the corresponding clean image was correctly
    classified by the source network.
    then, consider only adv_images whose clean versions have been classified correctly by both target(img_data) and source

    Args:
    -------
    saved_corr_images: An iterable object of dataset which yields all the images correctly classified by the target (Current net)

    adv_images: A .npz containing images that are adversarial in nature and are created at source net (npz should have image, labels, index)
    

    Returns:
    --------------
    An iterable dataset that contains adversarial images whose benign counter parts where classified correclty by both source and target network
    
    """
    print("Getting images that are correctly classified by both source and target(this) network...")
    
    adv_data = save_restore_images.save_or_load_image__npz("load", adv_images)
    #cpi_st = np.empty([0,28,28]) # create a empty numpy array to hold all the correctly predicted images from both source and target
    cpi_st = []
    correct_labels = []         # List containing all the correct labels (later change to numpy while saving to file)
    image_index = []            #image index of all the correctly predicted data
    total = 1000                #how adversarial images to take
   
    for img_data in saved_corr_images: 
        #corr_image = (img_data[0])[np.newaxis, :, :]
        corr_label = img_data[1]
        corr_image_index = img_data[2]        
        #compare the image index of both the source and the targe db.
        #please be aware that the "image-index" should always be from a source db like MNIST or CIFAR           
        if(corr_image_index in adv_data.image_index):
            position_image_index = np.where(adv_data.image_index == corr_image_index)            
            adv_image = adv_data.images[position_image_index][0] # you have to add [0] index here because
            #position_image_index or np_where is an array type becuse it expects that we find multiple images of this index
            # and so it is not just an int but a array and thus returning (1,28,28) because there can be multiple images)
               
            #cpi_st = np.append(cpi_st, adv_image, axis=0)
            cpi_st.append(adv_image) 
            list.append(correct_labels, corr_label)
            image_index.append(corr_image_index)
            #to test (data_test[corr_image_index] shoud give benign image of adv_data[position_image_index])
            #print("match{},{}", position_image_index, corr_image_index)
        if(len(image_index) >= Adv_sample_number_total):
            break
    if(len(cpi_st) < 1000):
        logger.info("Error: Less than 1000 correctly predicted images in both source and target")
    else:
        logger.info("1000 intersecting correct samples were found")
    
    cpi_st = np.array(cpi_st)
    savedir = logger.get_logger_dir() 
    save_prefiltered_images = save_restore_images.save_or_load_image__npz(
        "save", 
        os.path.join(savedir, "mnist_prefiltered_images.npz"),
        image= cpi_st, 
        labels=np.array(correct_labels), 
        image_index= np.array(image_index)) 
     
    return save_prefiltered_images


def create_attack(userconfig):
    """
        create attacks based on user config

        This function will use the below parameters (all form yml file) to craft adversarial examples on a defined net.

            1.  saved_model: the model checkpoint that has model parameters
        
            2.  base-model: model description of the model in which adv. images should be crafted
        
            3.  precision: precision of the model to be used

            4.  dataset: dataset to be used to get the images to craft adversarial examples on

            5.  algorithm: Attack algorithm that is to be used to craft adversarial examples

        At the end this will create a npz file that contains the adversarial examples, the correct label and the image index in the source databse
    """
    #get the saved model from disk
    saved_model = userconfig["attack-options"]["create-attack"]["load-model"]

    #get the dataset for creating attacks
    userdataset = userconfig["attack-options"]["create-attack"]["dataset"]
    dataset = get_data(userdataset)
    #the default dataset iterable object, in case of CIFAR, does not have object with images and lables like MNIST
    #so create another buffer object which has that and also is iterable
    if(userdataset == "cifar10"):
        dataset = getaugmenteddata_with_all_images(dataset)
        
    #the precision here should be a comma separated value viz: 2,2,32
    precision = userconfig["attack-options"]["create-attack"]["precision"]["bitwidth"]  
        
    #get the attack algorithm which will be used to craft adversarial attacks
    attack_algorithm_str = userconfig["attack-options"]["create-attack"]["algorithm"]
    algorithm = get_attack_algorithm(attack_algorithm_str)

        #get the base model description. The base model description should match the saved model description
    attack_model_description = userconfig["attack-options"]["create-attack"]["base-model"]  
        
    #setup logger
    dataset_name_log = userdataset
    if(precision is None):
        log_name = "FP"
    else:
        log_name = precision    
    logger.set_logger_dir(os.path.join('logs', 'trained_images', '{}-{}'.format(dataset_name_log, log_name)))
        
    #get model architecture
    #model = mr.get_model(mr.get_mod_typ(attack_model_description, precision), precision)
    
    if(userdataset == "mnist"):
        model = mr.get_model(mr.get_mod_typ(attack_model_description, precision), precision)
    elif(userdataset == "cifar10"):
        model = mr.get_model_cifar(mr.get_mod_typ_cifar(attack_model_description, precision), precision)

    #setup configurations for restoring model for predictions and for creating adversarial attacks
    config = Inference.inference_net.get_prediction_config(saved_model, model)

    #get data that are correctly classified by the classifier 
    _, _, saved_corr_images,_, _, _ =  Inference.inference_mode(config, dataset)
    #create adversarial attacks for the specified model
    create_attack_mode(config, algorithm, log_name, saved_corr_images, dataset)
    
def transfer_attack(userconfig):
    """
        use the setting from user config to get:

        Details of source:

            1. source images in npz format which are adversarial in nature

            2. source dataset from which the adversarial images were created

        Details of target:

            1. target model on which the adversarial attack should be performed

            2. target model graph

            3. saved checkpoint of target model

        All of this info will be used by this function to perform a transfer attack from the source to the target model.
        The target model here is only used for inferencing, this means that this is a black-box attack.
    
        TODO: The target-details should be swappable with a model hosted on cloud.

        The source could be anything, it can be a list of images, it can be a dataset. So by isolating creating adv examples
        and attacking models we are seperating crafting and attacking part which might be helpful later 
    """

    #get the saved model from disk
    #TODO: it is not so proper to read this way but sadly I dont have time :(
    saved_model = userconfig["attack-options"]["transfer-attack"]["target"]["target-model"]
    if(saved_model is None):
        print("Please provide a saved target model checkpoint.")
        return
    
    #get the base model description. The base model description should match the saved model description
    target_model_description = userconfig["attack-options"]["transfer-attack"]["target"]["target-base-model"]
    if(target_model_description is None):
        print("Please provide a saved base model. This is required to compute the model description.")
        return  

    #the precision here should be a comma separated value viz: 2,2,32
    precision = userconfig["attack-options"]["transfer-attack"]["target"]["precision"]["bitwidth"]  
    if(precision is None):
        print("Precision not specified in the yml file. Using a full precision network as target.")        

    #get the dataset for creating attacks
    userdataset = userconfig["attack-options"]["transfer-attack"]["source"]["source-data"]
    if(userdataset is None):
        print("Please provide source data from which the adversarial images were derived.")
        return    
        
    #get the adversarial images crafted at source
    adv_images = userconfig["attack-options"]["transfer-attack"]["source"]["source-images"]
    if(adv_images is None):
        print("Please provide adversarial images to make inference on the target machine.")
        return

    #setup logger
    dataset_name_log = userdataset
    if(precision is None):
        log_name = "FP"
    else:
        log_name = precision  
    source_log_name = os.path.splitext(os.path.basename(adv_images))[0]  
    logger.set_logger_dir(os.path.join('logs','transfer_log','from-{}-transfer-'.format(source_log_name), '{}-{}'.format(dataset_name_log, log_name)))

    initiate_attack_transfer(userdataset, target_model_description, precision, saved_model, adv_images)

def initiate_attack_transfer(userdataset, target_model_description, target_precision, target_saved_model, source_adv_images):
    """
    This will initiate attack tranfer from the source images to the target images

    Args:

    userdataset: dataset in which the attack was created (MNIST/ CIFAR)

    target_model_description: type of model (lenet5, model_a, ..)

    target_precision: precision of the target model

    target_saved_model: location of the saved model (checkpoint)

    source_adv_images: adversarial images created at source   
    
    """


    dataset = get_data(userdataset)  

    #get model architecture
    #model = mr.get_model(mr.get_mod_typ(target_model_description, target_precision), target_precision)

    #if the dataset is mnist get mnist models, if the dataset is cifar get cifar models
    if(userdataset == "mnist"):
        model = mr.get_model(mr.get_mod_typ(target_model_description, target_precision), target_precision)
    elif(userdataset == "cifar10"):
        model = mr.get_model_cifar(mr.get_mod_typ_cifar(target_model_description, target_precision), target_precision)
        
    #setup configurations for restoring model for predictions and for creating adversarial attacks
    config = Inference.inference_net.get_prediction_config(target_saved_model, model)

    #get data that are correctly classified by the classifier 
    _, _, saved_corr_images, saved_adv_false_images, _, _ =  Inference.inference_mode(config, dataset)
    
    #get images that are correctly classified by both source and target.
    pre_filtered_images = get_prefiltered_images_fast(saved_corr_images, source_adv_images)
    
    #use the prefiltered images to carryout the adversarial inference on the model
    adv_acc, transfer_rate, saved_adv_corr_images, saved_adv_false_images, agvl2, avglinf =  Inference.inference_mode(config, pre_filtered_images, True)
    return adv_acc, transfer_rate, saved_adv_corr_images, saved_adv_false_images, agvl2, avglinf
    


def initiate_attack_creation(userconfig):
    """
        Take the user configuration from the yaml file and creates adversarial examples on the specified model.

        To create adversarial examples this function will:

                    a.  Set up a predictor config that uses the current session to establish saved variables from checkpoint

                    b.  Set up model graph via the get_model method

                    c.  Perform inference using the setup model graph, variables and return accuracy (adversarial or normal)


        Args:
        -------------
        userconfig: A dictionary object retured from yaml data loader. This object should contain details like:

                    a.  What model should be used to create attacks (saved model)

                    b.  What graph to use (modelDesc)

                    c.  Is the model quantized?

                    d.  If it is a quantized model, then what bitwidth to use?

                    e.  What dataset to use? Dataset is a 


    """  
    if(userconfig["attack-options"]["attack-mode"].lower() == "create"):
        create_attack(userconfig)
    elif(userconfig["attack-options"]["attack-mode"].lower() == "transfer"):
        transfer_attack(userconfig)
    else:
        print("Attack-mode not supported. Please specify correct mode in the config.yml file")
 