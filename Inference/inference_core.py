from tensorpack.utils import logger
from six import with_metaclass
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import save_restore_images
import visualize_data
from tensorpack import *
#if i add this here it will call the base.py of the tensorpack installation location: C:\Python37\lib\site-packages\tensorpack\predict
from tensorpack.predict.base import *
#Get custom Mnist data extraction (default tensorflow has an addition dimention that can cause problems during inference)
from DataSets.mnist import GetMnist
#TODO: Extend this to also do inference on selected images which users can input
import os
import norm_distances as lpnorm

def infer_model_dataset(predictorConfig, dataset=None, labels=None, index= None):

    """
    Display top 10 predictions without true labels
    args: 
        prediction config
        dataset: dataset here is a set of images i.e a numpy array of images only (eg for mnist dataset is of shape (10000, 28,28))

    """

    #output labels of the MNIST classifier
    predictor = OfflinePredictor(predictorConfig)
    words = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    wrong = 1
    if(dataset is None):
        print("Please provide dataset to perform inference from.")
        return
    count = 0
    for i in range(len(dataset)):        
        img = (dataset[i])[np.newaxis, :, :] #for mnist
        #img = (dataset[i])[np.newaxis, :, :,:]
        print("True label", labels[i])
        #print(img.shape)
        output_adv = predictor(img)[0]
        prob_org= output_adv[0]
        #format the output
        ret2 = prob_org.argsort()[-10:][::-1]
        print(index[i])
        names = [words[i] for i in ret2]
        #print(f + ":")
        print(list(zip(names, prob_org[ret2]))) 
        count = count +1        
        if(labels[i] != ret2[0]):
            print("unmatch")             
            if(wrong > 1):
                break
            wrong = wrong + 1
    print("total images in this dataset: ", count)


def multiple_infer_model_dataset(predictorConfig, output_labels, dataset=None, get_adv_accuracy= False):
    print("called here")
    #output labels of the MNIST classifier
    predictor = OfflinePredictor(predictorConfig)
    words = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    if(dataset is None):
        print("Please provide dataset to perform inference from.")
        return
    #output = MultiProcessDatasetPredictor(predictorConfig, dataset, 1)
    
    dataset.reset_state()
    try:
        sz = len(dataset)
    except NotImplementedError:
        sz = 0
    with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
        for dp in dataset:
            print(dp[0])
            print(dp[1])
            break
        pbar.update()




def infer_model(predictorConfig, output_labels, dataset=None, get_adv_accuracy= False):
    """
    Perform evaluation on the model.

    args:
    ---------
    predictorConfig : Predictor configuration, required to create a predictor object

    output_labels : set of output_labels the machine classifies data to

    dataset : an iterable object containg [image, label] yields. Number of yields is equal to the number of images/labels in dataset
            for eg: check Mnist class
            In case of adversarial attacks where the data set contains adversarial images, this will yield [benign_image, label, adv_image]

    get_adv_accuracy: Set this bool true if you want to find adversarial accuracy as well. To find adversarial accuracy
                        the dataset should contain adversarial images data as well.
              
    """

    inference_batch_size = 1000

    #output labels of the MNIST classifier
    predictor = OfflinePredictor(predictorConfig)

    if(dataset is None):
        print("Please provide dataset to perform inference from.")
        return

    print("Computing accuracy of the network...")
    acc_t1, err_t1, saved_correct_images, saved_false_images, agvl2, avglinf = batch_compute_model_accuracy(predictor, output_labels, dataset, batch_size= inference_batch_size, get_adv_accuracy= get_adv_accuracy)
    if(get_adv_accuracy == False):
        logger.info("The overall accuracy of the model: {}, error: {} ".format(acc_t1, err_t1))
    else:
        logger.info("The adversarial accuracy of the model: {}, Adversarial Transfer Rate from source to target: {} ".format(acc_t1, err_t1))
    #show_top_predictions(predictor, output_labels, dataset)
    return acc_t1, err_t1, saved_correct_images, saved_false_images, agvl2, avglinf

def compute_model_adversarial_accuracy(predictor, output_labels, dataset):
    
    """
    compute the adversarial accuracy of the model. This means accuracy of the model against adversarial images created from the
    benign images that caused correct classification on the model.
    In other words, adversarial accuracy should always be computed against images that are correctly classified by the network
    
    args:
    -------
    
        Predcitor: predictor object
        
        output_labels: List of output labes the network classifies data to
        
        dataset: an iterable object containg [image, label] yields
    """
    
    #number of correct pred
    cpi_i = np.empty([0,28,28]) #create a empty numpy array to hold all the correctly predicted images
    cpi_l = np.empty([0]) #create a empty numpy array to hold the corresponding labels of correctly predicted images
    correct_labels = [] 
    wrong_labels = []
    wpi_i = np.empty([0,28,28]) #create a empty numpy array to hold all the correctly predicted images
    wpi_l = np.empty([0]) #create a empty numpy array to hold the corresponding labels of correctly predicted images

    adv_samples =  np.empty([0,28,28]) #  List of adv samples whose original image was correcly classified by the network
    adv_samples_w =  np.empty([0,28,28]) #  List of adv samples whose original image was incorrectly classified by the network
   
    correct_org = 0
    correct_adv = 0
    wrong_org = 0
    wrong_adv = 0
    #current image index
    data_element = 0
    sz = len(dataset)    
    print("Finding images that are classified correctly on the network...")
    with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
        for img_label in dataset:          
            img = (img_label[0])[np.newaxis, :, :]
            adv_image = (img_label[1])[np.newaxis, :, :]
            correct_label = img_label[2]      #correct label is a scalar value as provided by dataset iterator  
            #print(img.shape)
            output_org = predictor(img)[0]
            prob_org= output_org[0]            
            #arrage probabilites by high to low and load their index/position in ret
            ret_org = prob_org.argsort()[-10:][::-1]            
            if(ret_org[0] == correct_label):
                correct_org = correct_org + 1
                #create a numpy array of original images that were classified correclty by the classifier
                cpi_i = np.append(cpi_i, img, axis=0)
                #print(correct_label)
                list.append(correct_labels, correct_label)
                #create a nimpy array of all the adv images whose counter-part were correctly predicted
                adv_samples = np.append(adv_samples, adv_image, axis=0)
                #print("Correctly predicted: ", cpi_i.shape)
                #print("corresponding adversarial samples", adv_samples.shape)
                #print("correctly predicted images labels", correct_labels)
            else:
                wrong_org = wrong_org + 1
                #print(ret_org)
                names = [output_labels[i] for i in ret_org]
                #create a numpy array of original images that were not classified incorreclty by the classifier                
                wpi_i = np.append(wpi_i, img, axis=0)
                list.append(wrong_labels, correct_label)
                #create a nimpy array of all the adv images whose counter-part were incorrectly predicted
                #print(correct_label)
                adv_samples_w = np.append(adv_samples_w, adv_image, axis=0)
                #print("wrongly predicted: ", wpi_i.shape)
                #print("wrongly predicted images labels", wrong_labels)


            pbar.update()
        
        print("correctly predicted images labels", len(correct_labels))
        print("wrongly predicted images labels", len(wrong_labels))

        data_element = data_element + 1
        visualize_data.plot_images(wpi_i)
        
        #visualize_data.plot_images(adv_samples)

    sz = len(adv_samples)    
    print("Finding adversarial accuracy of the network...")
    #index = 0
    with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
        for index in range(len(adv_samples)):          
            img = (adv_samples[index])[np.newaxis, :, :]
            correct_label = correct_labels[index]      #correct label is a scalar value as provided by dataset iterator     
            #print(img.shape)
            output_org = predictor(img)[0]
            prob_org= output_org[0]            
            #arrage probabilites by high to low and load their index/position in ret
            ret_org = prob_org.argsort()[-10:][::-1]            
            if(ret_org[0] == correct_label):
                correct_adv = correct_adv + 1
                #print("actual label:", correct_label)
                #print("predicted", ret_org)
                #visualize_data.plot_images(img)
                #create a numpy array of original images that were classified correclty by the classifier
                #cpi_i = np.append(cpi_i, img, axis=0)
                #print(correct_label)
                #list.append(correct_labels, correct_label)
                #create a nimpy array of all the adv images whose counter-part were correctly predicted
                #adv_samples = np.append(adv_samples, adv_image, axis=0)
                #print("Correctly predicted: ", cpi_i.shape)
                #print("corresponding adversarial samples", adv_samples.shape)
                #print("correctly predicted images labels", correct_labels)
            else:
                wrong_adv = wrong_adv + 1
                #print(ret_org)
                #names = [output_labels[i] for i in ret_org]
                #create a numpy array of original images that were not classified incorreclty by the classifier                
                #wpi_i = np.append(wpi_i, img, axis=0)
                #list.append(wrong_labels, correct_label)
                #create a nimpy array of all the adv images whose counter-part were incorrectly predicted
                #print(correct_label)
                #adv_samples_w = np.append(adv_samples_w, adv_image, axis=0)
                #print("wrongly predicted: ", wpi_i.shape)
                #print("wrongly predicted images labels", wrong_labels)


            pbar.update()

    
    accuracy_t1 = (correct_org/len(dataset))*100    #top 1 accuracy for overall clean samples
    error_t1 = (wrong_org/len(dataset))*100         #top 1 error for all clean samples

    adv_accuracy_t1 = (correct_adv/len(adv_samples))*100    #top 1 accuracy for overall clean samples
    adv_error_t1 = (wrong_adv/len(adv_samples))*100         #top 1 error for all clean samples

    print("adv accuracy: ", adv_accuracy_t1)    
    print("adv error: ", adv_error_t1)
    return accuracy_t1, error_t1

def compute_model_accuracy(predictor, output_labels, dataset, get_adv_accuracy=False):
    """
    Compute model accuracy in a naive way. Just run the dataset through predictor and get the accuracy of the model.
    This is how keras does evaluation (model.eval())..
    May be there is a way to run this in a batch.

    This function will:

        1. Compute accuracy when get_adv_accuracy=False

        2. Compute adversarial accuracy and transfer rate when get_adv_accuracy=True

        3. ADDS "INDEX" ELEMENT TO THE DATASET IF THERE IS NO INDEX

    args:
    -------
    
         Predcitor: predictor object
        
         output_labels: List of output labes the network classifies data to
        
         dataset: an iterable object containg [image, label] yields

         get_adv_accuracy: true only when finding adversarial accuracy. The dataset in this case should only contain adversarial images, labels, index
    """
    cpi_i = np.empty([0,28,28]) # create a empty numpy array to hold all the correctly predicted images
    #cpi_i = []
    correct_labels = []         # List containing all the correct labels (later change to numpy while saving to file)

    wpi_i = np.empty([0,28,28]) #create a empty numpy array to hold all the correctly predicted images    
    wrong_labels = []           # List containing all the incorrect labels (later change to numpy while saving to file)
    #wpi_i = []

    image_index_correctly_predicted = [] #image index of data point that was correctly predited
    image_index_incorrectly_predicted = [] #image index of data point that was incorrectly predited

    correct = 0                 # number of correct predictions
    wrong = 0                   # number of wrong predictions
    #current image index
    data_element = 0    
    #when computing sadversarial accuracy (after getting all samples classified correctly) no need to savve the adv_component separately

    sz = len(dataset)
    with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
        for img_label in dataset:            
            img = (img_label[0])[np.newaxis, :, :]
            correct_label = img_label[1]
            #print(img.shape)
            output = predictor(img)[0]           
            prob= output[0] #by default tensorpack predictor returns output as an array when a model returns multiple output but in this case there is only one output
            #arrage probabilites by high to low and load their index/position in ret
            ret = prob.argsort()[-10:][::-1]
            if(ret[0] == correct_label):
                correct = correct + 1
                #print("Correct Label:", correct_label)
                #print(ret)
                #create a numpy array of original images that were classified correclty by the classifier
                cpi_i = np.append(cpi_i, img, axis=0)
                #cpi_i.append(img_label[0])
                #keep list of all corresponding labels of correctly predicted images                
                list.append(correct_labels, correct_label)
                #if there is no image index, add one
                if(len(img_label) == 2):           
                    image_index_correctly_predicted.append(data_element)
                else:
                    image_index_correctly_predicted.append(img_label[2])
                #print("current image number in dataset :", data_element)
            else:
                wrong = wrong + 1
                #print("correct label:", correct_label)
                #print ("prediction:")
                #print(ret)
                names = [output_labels[i] for i in ret]
                #print(list(zip(names, prob[ret])))
                #print("data element: ", data_element)
                #create a numpy array of original images that were not classified incorreclty by the classifier                
                wpi_i = np.append(wpi_i, img, axis=0)
                #wpi_i.append(img_label[0])
                list.append(wrong_labels, correct_label)

                #if there is no image index, add one. there is no image index means that there are only image, label in the dataset
                #for all other cases where the application itself creates adversarial examples, there is always index in it
                if(len(img_label) == 2):           
                    image_index_incorrectly_predicted.append(data_element)
                else:
                    image_index_incorrectly_predicted.append(img_label[2])
            #count = count + 1
            #if(count >10):
            #    break
            pbar.update()
            #which element of array gave wrong result. This is helpful for debugging    
            data_element = data_element + 1

    #cpi_i = np.array(cpi_i)
    #wpi_i = np.array(wpi_i)
    accuracy_t1 = (correct/len(dataset))
    error_t1 = (wrong/len(dataset))
    #print(cpi_i.shape)
    #o = predictor(cpi_i[0:10])[0]
    #print(o)
    #print(o.shape)
    #save all correctly and incorrectly predicted images separately    
    saved_correct_pred, saved_wrong_pred = save_files_to_disc(cpi_i, correct_labels, wpi_i, wrong_labels, image_index_correctly_predicted, image_index_incorrectly_predicted, get_adv_accuracy)
    return accuracy_t1, error_t1, saved_correct_pred, saved_wrong_pred


def batch_compute_model_accuracy(predictor, output_labels, dataset, batch_size=1000, get_adv_accuracy=False):
    """
    Compute model accuracy in a batches. Run the dataset through predictor and get the accuracy of the model.
    This is how keras does evaluation (model.eval())..    

    This function will:

        1. Compute accuracy when get_adv_accuracy=False

        2. Compute adversarial accuracy and transfer rate when get_adv_accuracy=True

        3. ADDS "INDEX" ELEMENT TO THE DATASET IF THERE IS NO INDEX

    args:
    -------
    
         Predcitor: predictor object
        
         output_labels: List of output labes the network classifies data to
        
         dataset: an iterable object containg [image, label] yields

         batch_size: specify the batch size in which inference should be done. Batch size should be less than the dataset size

         get_adv_accuracy: true only when finding adversarial accuracy. The dataset in this case should only contain adversarial images, labels, index
    """
   
    correct_labels = []         # List containing all the correct labels (later change to numpy while saving to file)
    cpi_i = []                  # create a empty numpy array to hold all the correctly predicted images
         
    wpi_i = []                  #create a empty numpy array to hold all the incorrectly predicted images
    wrong_labels = []           # List containing all the incorrect labels (later change to numpy while saving to file)

    image_index_correctly_predicted = [] #image index of data point that was correctly predited
    image_index_incorrectly_predicted = [] #image index of data point that was incorrectly predited

    correct = 0                 # number of correct predictions
    wrong = 0                   # number of wrong predictions
    #current image index
    #global index of image currently being processed. This is used as image index later 
    data_element = 0 
    
    #seperate data to batches.
    #The BatchData class will partition the given dataset into multiple batches. The iteratables (the elements returned as iterables) are preserved
    #data_test is then an iterable class that returns (batchnum, dataset_partition)
    data_test = BatchData(dataset, batch_size, remainder=True)
    
    sz = len(data_test)
    print("Number of batches created for {} data points in the dataset: {}".format(len(dataset), len(data_test)))
    with get_tqdm(total=sz, disable=(sz == 0)) as pbar:
        for element in data_test:
            #print(data_element)
            #index of elements in current batch.
            #data_element is gloal index while data_element_batch is local
            data_element_batch = 0
            image_batch = element[0]    #array containing batch of current images
            labels_batch = element[1]   #array containg batch of current labels
            #if the image has image index then take the image index batch as well for mapping later. array containing indexes of current batch
            if(len(element) >2):
                index_batch = element[2]
            output = predictor(image_batch)[0]
            #here the "output is a (batch,y) tensor. y being the probability of each of the output neurons"
            #for batchsize = 1000, the shape then is (1000,10), for 500 batch num the shape is (500,10)
            for result in output:
                #arrage probabilites by high to low and load their index/position in ret
                ret = result.argsort()[-10:][::-1]
                img = (image_batch[data_element_batch])
                if(ret[0] == labels_batch[data_element_batch]):
                    correct = correct + 1
                    #create a numpy array of images that were classified correclty by the classifier
                    #cpi_i = np.append(cpi_i, img, axis=0)
                    cpi_i.append(img)
                    list.append(correct_labels, labels_batch[data_element_batch])
                    #if there is no image index, add one. Index is added only during inference of the full dataset during normal inference.
                    #during inference of adversarial images, the index should already be there so the app does not add index there
                    if(len(element) == 2):   #len([0,2]) is 2, not zero based count        
                        image_index_correctly_predicted.append(data_element)
                    else:
                        image_index_correctly_predicted.append(index_batch[data_element_batch])
                else:
                    wrong = wrong + 1   
                    names = [output_labels[i] for i in ret]
                    #create a numpy array of original images that were not classified incorreclty by the classifier                
                    #wpi_i = np.append(wpi_i, img, axis=0)   
                    wpi_i.append(img)                
                    list.append(wrong_labels, labels_batch[data_element_batch])
                    #if there is no image index, add one. there is no image index means that there are only image, label in the dataset
                    #for all other cases where the application itself creates adversarial examples, there is always index in it
                    if(len(element) == 2):           
                        image_index_incorrectly_predicted.append(data_element)
                    else:
                        image_index_incorrectly_predicted.append(index_batch[data_element_batch])
                
                #increase global index by 1. 
                # Should be inside the inner loop because the index should be increased for each data point and not for only batches
                data_element = data_element + 1
                #batch index is reset every batch
                data_element_batch = data_element_batch + 1
            pbar.update()
    #get numpy arrays of the correctly and incorrectly predicted images
    cpi_i = np.array(cpi_i)
    wpi_i = np.array(wpi_i)
    #compute accuracy
    accuracy_t1 = (correct/len(dataset))
    error_t1 = (wrong/len(dataset))

    #save all correctly and incorrectly predicted images separately    
    saved_correct_pred, saved_wrong_pred = save_files_to_disc(cpi_i, correct_labels, wpi_i, wrong_labels, image_index_correctly_predicted, image_index_incorrectly_predicted, get_adv_accuracy)
    
    #get lp norm distances only during transfer (this should have been during only white box transfer but currently it calculates for all)
    #lp norm of only successful adversarial examples and hence during only adversarial transfer
    #for normal inferences, lp distances are zero so no use of this overhead during normal inference. 
    #Check only during adversarial transfer
    avglinf = 0 
    avgl2 = 0
    if (get_adv_accuracy):
        avgl2, avglinf = lpnorm.get_lp_norm_distances(wpi_i, image_index_incorrectly_predicted)
        logger.info("Average l2 distance of successful samples: {}, Average linf distance of successful samples: {}".format(avgl2, avglinf))
    return accuracy_t1, error_t1, saved_correct_pred, saved_wrong_pred, avgl2, avglinf

def save_files_to_disc(cpi_i, correct_labels, wpi_i, wrong_labels, image_index_correctly_predicted, image_index_incorrectly_predicted, get_adv_accuracy):    
    """
        Save the inference results to disk.

        This method will create two files in the directory where the logger is initialized.

            1. correct_pred_images/ correct_pred_adv_images: 
                contains images, labels and indices of the images that are correctly classified by the net

            2. incorrect_pred_images/ incorrect_pred_adv_images:
                contains image, labels and indices of the images that are incorrectly classified by the net

        further, the objects returned after saving are iterable and can be consumed if required

        Args:
        --------------

        cpi_i: correctly_predicted_images: a numpy array containing all the images that are CORRECTLY predicted

        correct_labels: Corresponding labels of all the images that were CORRECTLY predicted

        wpi_i: wrongly_predicted_images: a numpy array containing all the images that are INCORRECTLY predicted 

        wrong_labels: Corresponding labels of all the images that were INCORRECTLY predicted

        image_index_correctly_predicted: Corresponding indices of the images which are CORRECTLY predicted.
                                            It should be noted here that index here refers to index of image on the SOURCE db
                                            So, if the image is taken from MNIST the index here represents the index in MNIST dataset

        image_index_incorrectly_predicted: Corresponding indices of the images which are INCORRECTLY predicted.


        get_adv_accuracy: boolean used to control strings in logging and in console out. when true "adversarial" is added to prints

    """

    savedir = logger.get_logger_dir() 
    print(savedir)
    #If there is no adversarial component, adversarial image on the array no need to save it
    if(get_adv_accuracy):
        filename_corr = "correct_pred_adv_images.npz"
        filename_incorr = "incorrect_pred_adv_images.npz"
        image_type = " adversarial images "
    else:
        filename_corr = "correct_pred_images.npz"
        filename_incorr = "incorrect_pred_images.npz"
        image_type = " benign images "

    save_correct_pred = save_restore_images.save_or_load_image__npz(
            "save", 
            os.path.join(savedir, filename_corr), 
            image= cpi_i, 
            labels=np.array(correct_labels), 
            image_index= image_index_correctly_predicted)        
    logger.info("Total {} classified correctly by the network : {} ".format(image_type, cpi_i.shape))
    save_false_pred = save_restore_images.save_or_load_image__npz(
            "save", 
            os.path.join(savedir, filename_incorr), 
            image= wpi_i, 
            labels=np.array(wrong_labels), 
            image_index= image_index_incorrectly_predicted)       
    logger.info("Total {} classified incorrectly by the network : {} ".format(image_type, wpi_i.shape))    
    return save_correct_pred, save_false_pred

def show_top_predictions(predictor, output_labels, dataset):
    """
    This is a very baisc function meant just for debugging. It shows top 10 predictions from the model.
    
    To get any batch of predictions this should be modified   
    """
    start = 0
    end = 10
    for img_label in dataset:              
        img = (img_label[0])[np.newaxis, :, :]
        correct_label = img_label[1]
        print("correct label:", correct_label)
        #print(img.shape)
        output_adv = predictor(img)[0]
        prob_org= output_adv[0]
        #arrage probabilites by high to low and load their index/position in ret
        ret = prob_org.argsort()[-10:][::-1]
        names = [output_labels[i] for i in ret]
        #print(f + ":")
        #make predicted output_labels with the corresponding probability
        print(list(zip(names, prob_org[ret])))
        start = start +1
        if(start==end):
            break


def show_one_predictions(image, correct_label, predictor, output_labels): 
    """
    Give one image, along with correct label
    Get corresponding prediction.
    This is meant for debugging purposes.
    
    args:
    -------
        image: input image in numpy format
        
        correct_label: correct label of the image
        
        predictor: predictor object
        
        output_labels: set of output_labels the machine classifies data to
    """
    img = (image)[np.newaxis, :, :]         #for mnist
    #img = (image)[np.newaxis, :, :,:]
    print("correct label:", correct_label)
    #print(img.shape)
    output_adv = predictor(img)[0]
    prob_org= output_adv[0]
    #arrage probabilites by high to low and load their index/position in ret
    ret = prob_org.argsort()[-10:][::-1]
    names = [output_labels[i] for i in ret]
    #print(f + ":")
    #make predicted output_labels with the corresponding probability
    print("Predictions:")
    print(list(zip(names, prob_org[ret])))
    visualize_data.plot_image(image)
