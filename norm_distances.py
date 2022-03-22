from DataSets.mnist import GetMnist
from DataSets.cifar import Cifar10, get_cifar10_data, getaugmenteddata_with_all_images
import numpy as np

Mnist_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data"
Cifar_Data_Dir = r"C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\CIFAR10"


'''
Compute Lp distances between two images. 

Please refer to https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

for more info on how to compute these distances

'''
def get_lp_norm_distances(adversarial_images, image_index):
    """
    Get L2 and L_infinity distance between clean and adversarial images. Only the images which are successful are considered 
    
    Args
    -----
    adversarial images: a numpy array containing images [1000,28,28] means 1000 28x28 images
    
    image_index: Index of the images that were incorrectly classified/ successful adversarial images. 
                 Index is/(should be) from the source data (MNIST; CIFAR... not filtered data).
    
    """
    total_l2 = 0
    total_linf = 0
    total_l0 = 0


    #if the image_index does not have anything, means that none of the images were classified incorrectly. 
    # In this case there are no adv examples so the l2 and linf distance both are zero
    if(len(image_index)<= 0):  
         return 0,0

    # The dimension of collection fo adv images in case of mnist if (1000,28,28) but for CIFAR one additional dim makes 
    # it (1000,30,30,3)
    if(adversarial_images.ndim == 3):
        data_test = GetMnist('test', dir=Mnist_Data_Dir)
    else:
        data_test = get_cifar10_data("test", dir=Cifar_Data_Dir)
        data_test = getaugmenteddata_with_all_images(data_test)
    sourceimages = data_test.images
    j=0
    print(len(image_index))
    for index in image_index:
        #print(index)
        total_l2 = total_l2 + np.linalg.norm(adversarial_images[j] - sourceimages[index])         
        #print(np.linalg.norm(adversarial_images[j] - sourceimages[index]))
        total_linf = total_linf + np.max(abs(adversarial_images[j] - sourceimages[index]))
        #total_l0 = total_l0 + np.linalg.norm(adversarial_images[j].flatten() - sourceimages[index].flatten(), ord= 0) 
        j = j +1
    avg_l2 = total_l2/len(image_index)
    avg_linf = total_linf/len(image_index)
    #avg_l0 = total_l0/len(image_index)
    return avg_l2, avg_linf#, avg_l0
