# -*- coding: utf-8 -*-
# File: mnist.py

#Description: This class is the overridden function of from tensorpack/dataflow/dataset/mnist because there is no need to add
#additional dimension when extracting images in the extract_images

import gzip
import numpy
import os

from tensorpack.dataflow import dataset
from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.dataflow.dataset.mnist import Mnist, maybe_download, extract_labels, _read32


def extract_images(filename):
    print("extracting images...")
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        #there is no use in adding an additional dimension here as the last one is removed during queue during training.
        #if additional dim is required add it during inference
        #data = data.reshape(num_images, rows, cols, 1)
        data = data.reshape(num_images, rows, cols)
        data = data.astype('float32') / 255.0
        print("data shape is...", data.shape)
        return data

class GetMnist(Mnist):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    _DIR_NAME = 'mnist_data'
    _SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    # EDIT: Don't shuffle data by default
    def __init__(self, train_or_test, shuffle=False, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        # Dont download as the site is sometimes unresponsive. Too many ppl downloading. Instead use the local dataset
        print("this is a custom class")
        print("Shuffle is:", shuffle)
        #dir = r'C:\Users\sab\Downloads\AI Testing\_Tools\DataSets\MNIST\Data'
        if dir is None:
            dir = get_dataset_path(self._DIR_NAME)
        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        def get_images_and_labels(image_file, label_file):
            f = maybe_download(self._SOURCE_URL + image_file, dir)
            images = extract_images(f)
            f = maybe_download(self._SOURCE_URL + label_file, dir)
            labels = extract_labels(f)
            assert images.shape[0] == labels.shape[0]
            return images, labels

        if self.train_or_test == 'train':
            self.images, self.labels = get_images_and_labels(
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz')
        else:
            self.images, self.labels = get_images_and_labels(
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz')

    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
           print("shuffle is on")
           self.rng.shuffle(idxs)
        for k in idxs:
            #print("itter")
            img = self.images[k].reshape((28, 28))
            label = self.labels[k]
            yield [img, label]