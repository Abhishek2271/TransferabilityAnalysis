# -*- coding: utf-8 -*-
# File: cifar.py

#         Yukun Chen <cykustc@gmail.com>


##THIS IS AN EXACT COPY OF THE CIFAR READER IN THE TENSORPACK DATAFLOW BUT IT IS HERE BECAUSE I NEED TO PUT COMMENTS ON
# IT JUST FOR MY UNDERSTANDING. BUT; THIS FILE IS NOT COMPLETELY INTERCHANGABLE WITHT HE AFOREMENTIONED FILE.
#THERE ARE SOME MODIFICATION LIKE IMAGE NORMALIZATION AND DATA PREPRATION FOR TRAINING IS DONE HERE


import numpy as np
import os
import pickle
import tarfile

from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.dataflow.base import RNGDataFlow

from tensorpack.dataflow import imgaug  
from tensorpack.dataflow.image import AugmentImageComponent

__all__ = ['CifarBase', 'Cifar10', 'Cifar100']


DATA_URL_CIFAR_10 = ('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 170498071)
DATA_URL_CIFAR_100 = ('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 169001437)


def maybe_download_and_extract(dest_directory, cifar_classnum):
    """Download and extract the tarball from Alex's website. Copied from tensorflow example """
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        cifar_foldername = 'cifar-10-batches-py'
    else:
        cifar_foldername = 'cifar-100-python'
    if os.path.isdir(os.path.join(dest_directory, cifar_foldername)):
        logger.info("Found cifar{} datas in {}.".format(cifar_classnum, dest_directory))
        return
    else:
        DATA_URL = DATA_URL_CIFAR_10 if cifar_classnum == 10 else DATA_URL_CIFAR_100
        filename = DATA_URL[0].split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        download(DATA_URL[0], dest_directory, expect_size=DATA_URL[1])
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_cifar(filenames, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    ret = []
    for fname in filenames:
        fo = open(fname, 'rb')
        dic = pickle.load(fo, encoding='bytes')
        data = dic[b'data']
        if cifar_classnum == 10:
            label = dic[b'labels']
            IMG_NUM = 10000  # cifar10 data are split into blocks of 10000
        else:
            label = dic[b'fine_labels']
            IMG_NUM = 50000 if 'train' in fname else 10000
        fo.close()
        for k in range(IMG_NUM):
            img = data[k].reshape(3, 32, 32)
            img = np.transpose(img, [1, 2, 0])
            #normalize image when getting the image itself
            img = normalize_image(img)          
            ret.append([img, label[k]])
    return ret

def normalize_image(image):
    normalize_image = image.astype("float32")/255.0
    return normalize_image

def get_filenames(dir, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        train_files = [os.path.join(
            dir, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)]
        test_files = [os.path.join(
            dir, 'cifar-10-batches-py', 'test_batch')]
        meta_file = os.path.join(dir, 'cifar-10-batches-py', 'batches.meta')
    elif cifar_classnum == 100:
        train_files = [os.path.join(dir, 'cifar-100-python', 'train')]
        test_files = [os.path.join(dir, 'cifar-100-python', 'test')]
        meta_file = os.path.join(dir, 'cifar-100-python', 'meta')
    return train_files, test_files, meta_file


def _parse_meta(filename, cifar_classnum):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj['label_names' if cifar_classnum == 10 else 'fine_label_names']


#the data in the end is in NHWC format. the "data" variable contains the data and the corresponding label
class CifarBase(RNGDataFlow):
    """
    Produces [image, label] in Cifar10/100 dataset,
    image is 32x32x3 in the range [0,255].
    label is an int.
    """
    def __init__(self, train_or_test, shuffle=None, dir=None, cifar_classnum=10):
        """
        Args:
            train_or_test (str): 'train' or 'test'
            shuffle (bool): defaults to True for training set.
            dir (str): path to the dataset directory
            cifar_classnum (int): 10 or 100
        """
        assert train_or_test in ['train', 'test']
        assert cifar_classnum == 10 or cifar_classnum == 100
        self.cifar_classnum = cifar_classnum
        if dir is None:
            dir = get_dataset_path('cifar{}_data'.format(cifar_classnum))
        #here the extraction is done only when it is downloaded.
        #so extract the file manually and the "dir" then is the directory name where the tar file is
        maybe_download_and_extract(dir, self.cifar_classnum)
        #get training data, test data and the metafile
        #meta file contains mapping of labels with the label names
        #http://www.cs.toronto.edu/~kriz/cifar.html
        train_files, test_files, meta_file = get_filenames(dir, cifar_classnum)
        if train_or_test == 'train':
            self.fs = train_files
        else:
            self.fs = test_files
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self._label_names = _parse_meta(meta_file, cifar_classnum)
        self.train_or_test = train_or_test
        #data contains a 2d array each element containing [data, label]
        self.data = read_cifar(self.fs, cifar_classnum)     
        self.dir = dir

        if shuffle is None:
            shuffle = train_or_test == 'train'
        self.shuffle = shuffle

    def __len__(self):
        return 50000 if self.train_or_test == 'train' else 10000

    def __iter__(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            print("shuffle is on") #data is shuffled every epochs
            self.rng.shuffle(idxs)
        for k in idxs:
            # since cifar is quite small, just do it for safety
            yield self.data[k]

    def get_per_pixel_mean(self, names=('train', 'test')):
        """
        Args:
            names (tuple[str]): the names ('train' or 'test') of the datasets

        Returns:
            a mean image of all images in the given datasets, with size 32x32x3
        """
        for name in names:
            assert name in ['train', 'test'], name
        train_files, test_files, _ = get_filenames(self.dir, self.cifar_classnum)
        all_files = []
        if 'train' in names:
            all_files.extend(train_files)
        if 'test' in names:
            all_files.extend(test_files)
        all_imgs = [x[0] for x in read_cifar(all_files, self.cifar_classnum)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

    def get_label_names(self):
        """
        Returns:
            [str]: name of each class.
        """
        return self._label_names

    def get_per_channel_mean(self, names=('train', 'test')):
        """
        Args:
            names (tuple[str]): the names ('train' or 'test') of the datasets

        Returns:
            An array of three values as mean of each channel, for all images in the given datasets.
        """
        mean = self.get_per_pixel_mean(names)
        return np.mean(mean, axis=(0, 1))


class Cifar10(CifarBase):
    """
    Produces [image, label] in Cifar10 dataset,
    >> OLD--image is 32x32x3 in the range [0,255].
    >> NEW--image is normalized in the range [0, 1]
    label is an int.
    """
    def __init__(self, train_or_test, shuffle=None, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset, default to shuffle in training
        """
        print("shuffle is set to: ", shuffle)
        super(Cifar10, self).__init__(train_or_test, shuffle, dir, 10)


def get_cifar10_data(train_or_test, dir=None):
    """
    From the specified directory location, gets the CIFAR data.
    The returned dataset object is iterable and yields [image, label] pair. Label is an int
    """
    isTrain = train_or_test == 'train'
    #if cifar_classnum == 10:
    data_set = Cifar10(train_or_test, dir=dir)
    if isTrain:
        print("applying augments to training samples...")
        augmentors = [
            #imgaug.RandomCrop((30, 30)),
            #during training, flip the image with a probability of 50%. This is used as a countermeasure for overfitting
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(63),
            #imgaug.Contrast((0.2, 1.8)),
            #all the augmentation above change image visuals while the normalization will change image to float.
            #in MNIST for normalization we just divided each by 255.0 but here other factors are taken into account
            #imgaug.MeanVarianceNormalize(all_channel=True)
            #Normalize image to [0,1] range. The MeanVarianceNormalization I could not fully understand.
            #Let's see if this normalization works for the net
            #imgaug.MinMaxNormalize(min=0, max=1,all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            #imgaug.MeanVarianceNormalize(all_channel=True)
            #imgaug.MinMaxNormalize(min=0, max=1,all_channel=True)
        ]
    if(isTrain):
        data_set = AugmentImageComponent(data_set, augmentors)    
    #ds = BatchData(ds, 128, remainder=not isTrain)
    #if isTrain:
        #ds = MultiProcessRunnerZMQ(ds, 5)
    return data_set

class getaugmenteddata_with_all_images():

    """
    A really bad intrepretaion of dataset and I do not like it at all. But i dont have time to make it better.
    The augmented image object does not have array containing all images and all labels which means I have to get them
        seperately. An easy way would be to just not augment or reshape image to 30x30 but then the network accuracy is trrible (83%)
   
    """

    """
    DO NOT USE THIS IN CASE OF "train" dataset !!! because the image augmentator applies augmentations during iteration, in iterator
    This means that the random crop and random flip done in training will cause the data to be different in each iteration
    
    """
    def __init__(self, data_set):
        self.data_set = data_set
        self.images = []
        self.labels = []
        self.image_index = []
        for im_lbl in data_set:
            self.images.append(im_lbl[0])
            self.labels.append(im_lbl[1])
            if(len(im_lbl) >2):
                self.image_index.append(im_lbl[2])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.image_index = np.array(self.image_index)

    def __len__(self):
        return len(self.images)

    def __iter__(self):        
        for k in self.data_set:
            yield k


class Cifar100(CifarBase):
    """ Similar to Cifar10"""
    def __init__(self, train_or_test, shuffle=None, dir=None):
        super(Cifar100, self).__init__(train_or_test, shuffle, dir, 100)


if __name__ == '__main__':
    ds = Cifar10('train')
    mean = ds.get_per_channel_mean()
    print(mean)

    import cv2
    ds.reset_state()
    for i, dp in enumerate(ds):
        if i == 100:
            break
        img = dp[0]
        cv2.imwrite("{:04d}.jpg".format(i), img)
