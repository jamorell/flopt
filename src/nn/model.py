
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import random

import logging

from jsd_utils import _get_myrandom




def create_loss(name):
  if name == "CategoricalCrossentropy":
    return tf.keras.losses.CategoricalCrossentropy()



def create_optimizer(name, learning_rate):
  optimizer = None
  if name == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif name == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  return optimizer

def create_topology(name):
  m = None
  if name == "CONV":
    m = create_topology_mnist_conv_28_28_1()
  elif name == "DENSE":
    m = create_topology_mnist_dense_28_28_1()
  return m

def create_topology_mnist_conv_28_28_1():
  random_state = random.getstate()
  nprandom_state = np.random.get_state()
  random.seed(1)
  myrandom_seed = _get_myrandom()
  myrandom_seed(1) #tf.compat.v1.random.set_random_seed(1) #tf.random.set_random_seed(1)
  model = Sequential()

  model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                   activation ='relu', input_shape = (28,28,1)))
  model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                   activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                   activation ='relu'))
  model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                   activation ='relu'))
  model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = "softmax"))

  random.setstate(random_state)
  np.random.set_state(nprandom_state)
  return model


def create_topology_mnist_dense_28_28_1():
  random_state = random.getstate()
  nprandom_state = np.random.get_state()
  random.seed(1)
  myrandom_seed = _get_myrandom()  
  myrandom_seed(1) #tf.compat.v1.random.set_random_seed(1) #tf.random.set_random_seed(1)
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28, 1)))
  model.add(Dense(42, activation='relu'))
  model.add(Dense(10, activation = "softmax"))
  random.setstate(random_state)
  np.random.set_state(nprandom_state)
  return model
  
  
  


