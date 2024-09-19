import logging

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

import os
import gzip

import random

###DATA##########################################################
def loader_fashion():

  ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
  path_folder = ROOT_DIR + "/datasets/fashion_mnist/"
  print(path_folder)

  files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
  paths = []

  with gzip.open(path_folder + files[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(path_folder + files[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(path_folder + files[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(path_folder + files[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)


  ######
  num_classes = 10  
  theshape = [-1, 28, 28, 1]
  norm_x_train = x_train.astype("float32") / 255 
  norm_x_test = x_test.astype("float32") / 255
  encoded_y_train = to_categorical(y_train, num_classes=num_classes, dtype="float32")
  encoded_y_test = to_categorical(y_test, num_classes=num_classes, dtype="float32")
  X_train = norm_x_train.reshape(theshape) 
  Y_train = encoded_y_train
  X_test = norm_x_test.reshape(theshape) 
  Y_test = encoded_y_test
  ######
  
  #print(x_train)
  #return (x_train, y_train), (x_test, y_test)
  #return x_train, y_train, x_test, y_test
  return X_train, Y_train, X_test, Y_test
  


def loader_mnist():
  ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
  path_folder = ROOT_DIR + "/datasets/mnist/"
  (x_train, y_train), (x_test, y_test) = load_data(path_folder + '/mnist.npz')
  num_classes = 10
  theshape = [-1, 28, 28, 1]
  #logging.info(x_train.shape)
  #logging.info(y_train.shape)

  norm_x_train = x_train.astype("float32") / 255 
  norm_x_test = x_test.astype("float32") / 255


  encoded_y_train = to_categorical(y_train, num_classes=num_classes, dtype="float32")
  encoded_y_test = to_categorical(y_test, num_classes=num_classes, dtype="float32")

  X_train = norm_x_train.reshape(theshape) #norm_x_train.reshape(-1, 28, 28, 1)
  Y_train = encoded_y_train
  X_test = norm_x_test.reshape(theshape) #norm_x_test.reshape(-1, 28, 28, 1)
  Y_test = encoded_y_test
  return X_train, Y_train, X_test, Y_test
  
  
  
def divide_data_between(X_train, y_train, n_devices):
  indexes = np.array(list(range(len(X_train))))
  random.shuffle(indexes)
  Xs_train = []
  ys_train = []
  n_elem_per_d = len(X_train) / n_devices
  for i in range(n_devices):
    first = int(i * n_elem_per_d)
    last = int(first + n_elem_per_d)
    local_indexes = indexes[first: last]
    local_X = X_train.take(local_indexes, axis = 0)

    local_Y = y_train.take(local_indexes, axis = 0)
    
    Xs_train.append(local_X)
    ys_train.append(local_Y)
    
  return Xs_train, ys_train
