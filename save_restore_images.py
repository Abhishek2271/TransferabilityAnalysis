import numpy as np
import cv2
import os
from tensorpack.dataflow.dataset.mnist import Mnist 

'''
    The classes/ functions here are meant to load/ save images in numpy and png format as per requirement. There are different ways to save
        and restore images. Below are few that might be helpful. This class only implements:
            1. Saving/restoring as numpy arrays
            2. Saving/restoring with cv2         
    Ways to save an image:
    1. As numpy array with np.save then restoring as np.load
        lossless and can save multiple images in a same file.

    2. Saving as an image using CV2
        a. Can save numpy array as an image
        b. Load the saved image
        
        Need data to be in uint8 format this means converting the float values to utf8 to save and from uint8 to float back to read.
        
        But denormalizing > save and then load > normalize also give same numpy array
        
        By default everything is in RGB so no need load in greyscale

    3. Saving as an image with PIL. 
        a. Can save numpy array as an image
        b. Load the saved image.
        
        Need data to be in uint8 format this means converting the float values to utf8 to save and from uint8 to float back to read.
        
        But denormalizing > save and then load > normalize also give same numpy array
        
	By default everything is in greyscale so no need to do anything for MNIST
    
    4. Saving with plt.
        a. Can save numpy array as image
        b. Load the saved image
        We can do this but the problem is that matplt save the entire plot and not individual images. So to save individual images we again need to do another loop.
        Also defining plt we have to be precise on the plot
        
    NOTE:      

    Saving as TIFF
        Saving as tiff file might have obvious benefits but also pitfalls
        a. No need for normalize and denomalizing which is a benefit BUT
        b. But preferable use CV2 or PIL because then we are sure what format (uint8) we are reading so we can normalize later when it is required. 
            Also while writing PIL gives error when the data is in float32 so it is also a way to know if data was normalized before.
            Similarily, CV2 does not give error but it saves a blank image or incorrect image when data is in float32 so again a way to know the data is normalized.
'''
def save_numpy_array(image_array, filename):
    """
    Saves imageset in .npy format 

    args        
    --------
    image_array: numpy array of images to save

    filename: the filename (location) to which the npy file will be saved
    """
    np.save(filename, image_array)


def load_numpy_array(filename):
    """
    loads imageset saved in .npy format

    args        
    --------

    filename: the filename (location) from which the npy file will be loaded
    """
    return(np.load(filename))

def save_image(images, dir, image_denormalized, labels=None):    
    '''
    Save image to png format using OpenCV

    args
    ---------------
    images : dataset containing numpy array of images

    dir : directory location to save the image

    image_denormalized : True if the image is denormalized, False if the image need denormalization

    labels : provide data labels corresponding to images. Images will be saved as <label_name>.png, 
            when not given, images will be named sequentially in the order they appear on the dataset

    '''
    if(image_denormalized == False):
        denorm_images = denormalize_image(images)
    image_name = labels
    label_index = 0  # use the label index as name when labels are not provided
    for im in denorm_images:
        if (image_name is None):
            save_path = os.path.join(dir, (str)(label_index)+ ".png")
        else:
            save_path = os.path.join(dir, (str)(labels[label_index])+ ".png")
            print(save_path)

        #need to denormalize before saving the image, if the image is not denormalized before        
        cv2.imwrite(save_path, im)
        label_index = label_index + 1    #cv2.imwrite(r'C:\tmp\SaveImages\{}.png'.format(labels), denormalized_train[1])


def load_image(filename, return_normalized, isGreyScale=True): 
    '''
    Load images  from png format using OpenCV

    args:
        filename: filename of the image to load
        dir: directory location to load the image
        return_normalized: CSV by default returns non-normalized images i.e. uint8 [0,255] 
                            when this bool is true the function will return normalized images ie float32 [0,1]
        isGreyScale:  cv2 returns RGB by default for greyscale this need to be specified

    returns:
        The image in a numpy format and its label
    '''
    #TODO: need to get label in a more dependable way. Not like this lol.
    label = (os.path.basename(filename)).split(".")[0] #get the label as the name of the file      
    if(isGreyScale):
        image_arr = np.array(cv2.imread(filename, 0))
    else:
        image_arr = np.array(cv2.imread(filename))
    #if image needs to be normalized then normalize it. Dont do this automatically. 
    if (return_normalized):
        image_arr = normalize_images(image_arr)

    return image_arr, label
    #check if the image integrity is maintained
    #a = np.array_equal(image_arr,data.images[1])
    #print(a)

def denormalize_image(imageset):
    denormalized_images = (imageset * 255).astype("uint8")
    return denormalized_images

def normalize_images(imageset):
    normalize_images = imageset.astype("float32")/255.0
    return normalize_images


class save_or_load_image__npz():
    """
    save or Load images from npz file and make it iterable returning iterations over each image, label pair on the dataset.
    This should produce data in the same vain as the Datasets\Mnist Class of the tensorpack
    
    constructor arguments:
    ---------------------

    saveorload : "save" for saving the file
                 "load" for loading the file

    filelocation : file location from where the images and label values are to be loaded

    image : image to save as a numpy array (pass nothing when loading)

    labels : image labels as numpy array (pass nothing when loading)

    """
    def __init__(self, saveorload,  filelocation, image = None, labels=None, image_index=None):
        if(saveorload == "save"):    
            self.images = image
            self.labels = labels
            self.image_index = image_index
            self.save_npz_array(image, labels, image_index, filelocation)            
        elif(saveorload == "load"):
            self.loaded = np.load(filelocation)
            self.images = self.loaded["images"]
            self.labels = self.loaded["labels"] 
            self.image_index = self.loaded["image_index"]

    def save_npz_array(self, image, labels, image_index, filename):   

        """
            Saves images as well as labels in .npz format

            args
            -------
            
            image : numpy array of images to save

            labels : correspoinding labels

            filename : the filename (location) to which the npy file will be saved
        """ 
        self.images = image
        self.labels = labels
        np.savez_compressed(filename, images=image, labels=labels, image_index= image_index)

    #adding length is always a good idea but not sure if this is really useful ...
    def __len__(self):
        return (self.images.shape[0])    

    #return a iterator for the data within the restored npz file
    def __iter__(self):
        _len = list(range(self.__len__()))
        for i in _len:
            yield [self.images[i], self.labels[i], self.image_index[i]]  


class save_or_load_adversarial_image__npz():
    """
    save or Load images from npz file and make it iterable returning iterations over each image, label pair on the dataset.
    This should produce data in the same vain as the Datasets\Mnist Class of the tensorpack
    
    constructor arguments:
    ---------------------

    saveorload : "save" for saving the file
                 "load" for loading the file

    filelocation : file location from where the images and label values are to be loaded

    image : image to save as a numpy array (pass nothing when loading)

    labels : image labels as numpy array (pass nothing when loading)

    """
    def __init__(self, saveorload,  filelocation, benign_image = None,  labels=None, adversarial_image = None, image_index=None):
        #set true if the loaded image file has adversarial image data on it.
        self.hasadversarial = False 
        
        if(saveorload == "save"):    
            self.benign_images = benign_image
            self.adversarial_images = adversarial_image
            self.labels = labels
            self.image_index = image_index
            self.save_npz_array(self.benign_images, self.labels, self.adversarial_images, self.image_index, filelocation)            
        elif(saveorload == "load"):
            self.loaded = np.load(filelocation)
            self.benign_images = self.loaded["benign_images"]
            if("adversarial_images" in self.loaded):
                self.adversarial_images = self.loaded["adversarial_images"]
                self.hasadversarial = True
            else:
                self.hasadversarial =False
            self.labels = self.loaded["labels"] 
            self.image_index = self.loaded["image_index"]

    def save_npz_array(self, benign_image, labels, adversarial_image, image_index, filename):   
        
        """
            Saves images as well as labels in .npz format

            args
            -------
            
            image : numpy array of images to save

            labels : correspoinding labels

            filename : the filename (location) to which the npy file will be saved
        """ 
        self.benign_images = benign_image
        self.adversarial_images = adversarial_image
        self.labels = labels
        self.image_index = image_index
        if(adversarial_image is None):
            np.savez_compressed(filename, benign_images=self.benign_images, labels=self.labels, image_index= self.image_index)
            self.hasadversarial = False           #this is set here because sometimes we may save the image and need to use the object
        else:
            np.savez_compressed(filename, benign_images=self.benign_images, labels=self.labels, image_index= self.image_index, adversarial_images= self.adversarial_images)        
            self.hasadversarial = True
    #adding length is always a good idea but not sure if this is really useful ...
    def __len__(self):
        return (self.benign_images.shape[0])    

    #return a iterator for the data within the restored npz file
    def __iter__(self):
        _len = list(range(self.__len__()))
        for i in _len:
            #if the loaded numpy array has adversarial image data then return adversarial images as a component.
            if(self.hasadversarial):
                yield [self.benign_images[i], self.labels[i], self.image_index[i], self.adversarial_images[i]]
            else:
                yield [self.benign_images[i], self.labels[i], self.image_index[i]]  

if __name__ == '__main__':
    data = Mnist("test", dir=r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data")
    #save_image(data.images[:10], r"c:\tmp\SaveImages", False, data.labels)
    #load_image(r"C:\tmp\SaveImages\7.png", True)
    npz_image = save_or_load_adversarial_image__npz("load", r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\TrainedImages\mnist_conv_full_pre.npz")
    #npz_image = save_or_load_adversarial_image__npz("save", r"C:\Users\sab\Downloads\AI Testing\Source\Dorefanet\tensorpack\FullPrecisionModels\TrainedImages\mnist_conv_full_pre.npz", data.images[:100], data.labels[:100], data.images[:100])
    
    print(npz_image.benign_images.shape)
    print(npz_image.hasadversarial)
    print(npz_image.adversarial_images.shape)
    print(npz_image.labels.shape)
    index = 0
    #for i in npz_image:        
        #print(np.array_equal(i[0], data.images[index]))
        #index = index + 1
        #print(i["benign_images"].shape)
        #print(i[2].shape)
    
     