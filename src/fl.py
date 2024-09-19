import numpy as np
import random

import tensorflow as tf


#'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)    
    
#tf.config.experimental.set_virtual_device_configuration(
#        gpu_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])    

gpu_memory_fraction = 0.1 # Choose this number through trial and error
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
#tf.config.experimental.set_lms_enabled(True)
#'''

from federated_learning.agg_techinques import *

import os

#disable warnings
try:
  import tensorflow.python.util.deprecation as deprecation
  deprecation._PRINT_DEPRECATION_WARNINGS = False
except:
  pass
#


import logging

import json
#from ml_metrics import kappa
from sklearn.metrics import cohen_kappa_score



from jsd_utils import _get_myrandom
from jsd_utils import *
from nn.jsd_nn_utils import *
from actors.worker import JSDWorker
from actors.aggregator import JSDAggregator

from data import *

#mypercentile = None
myrandom_seed = _get_myrandom() #None
myquantize_and_dequantize = None



myquantize_and_dequantize = tf.quantization.quantize_and_dequantize_v2



def mytrain(worker, aggregator):
  logging.info("worker.assign(aggregator.model.trainable_variables)")
  worker.assign(aggregator.model.trainable_variables)  
  print("worker.train()")   
  worker.train()  

class JSDEvaluate():
  def __init__(self, pp):
    if pp.dataset == pp.DATASET_TYPE.MNIST:
      self.X_train, self.y_train, self.X_test, self.y_test = loader_mnist()
    elif pp.dataset == pp.DATASET_TYPE.FASHION:
      self.X_train, self.y_train, self.X_test, self.y_test = loader_fashion()
    else:
      raise Exception("Selected dataset does not exists! ")          
           
  def evaluate(self, solution, pp, nn_seed = 1): ##pp = problem_parameters
    """
    # solution is organised as
        precision for each layer
        number of slaves to communicate
        number of training steps
        threshold for each layer
    """   
    ##################
    solution = list(map(round, solution)) #solution = list(map(int, solution))
    chosen_topology = pp.chosen_topology
    n_devices = pp.n_devices
    n_layers = pp.n_layers
    speeds = pp.speeds 
    
    
    gradients_or_weights = pp.get_param(pp.PARAM_NAME.WEIGHTS_OR_GRADIENTS, solution)
    if gradients_or_weights == TYPE_WEIGHTS_OR_GRADIENTS.GRADIENTS:
      is_storing_gradients = True
    elif gradients_or_weights == TYPE_WEIGHTS_OR_GRADIENTS.WEIGHTS:
      is_storing_gradients = False
    else:
      raise Exception("Gradients or weights not configured! " + key)  
      
    #is_storing_gradients = pp.get_param(pp.PARAM_NAME.WEIGHTS_OR_GRADIENTS, solution) #True # TODO
    ratio_type = pp.get_param(pp.PARAM_NAME.RATIO, solution)# = RatioType.NORMAL # TODO -> NSGA-II must select this
    zeros_technique = pp.get_param(pp.PARAM_NAME.ZEROS_TECHNIQUE, solution) #TYPE_ZEROS_TECHNIQUE.SKIP # TODO -> NSGA-II must select this
    m_batch_size = pp.get_param(pp.PARAM_NAME.M_BATCH_SIZE, solution)
    incremental_max_local_steps = pp.get_param(pp.PARAM_NAME.INCREMENTAL_MAX_LOCAL_STEPS, solution)
    
    logging.info("n_layers = " + str(n_layers))
    logging.info("len(solution) = " + str(len(solution)))
    logging.info("solution = " + str(solution))
    #exit()
    
    quantization_precission = pp.get_param(pp.PARAM_NAME.QUANTIZATION, solution)
    local_steps = pp.get_param(pp.PARAM_NAME.LOCAL_STEPS, solution)
    threshold_percentage = pp.get_param(pp.PARAM_NAME.SPARSIFICATION, solution)
    threshold_percentage = np.array(threshold_percentage).astype(np.float)
    selected_workers_per_aggregation = pp.get_param(pp.PARAM_NAME.SELECTED_DEVICES, solution)  

    
    print(quantization_precission)
    print(selected_workers_per_aggregation)
    print(local_steps)
    print(threshold_percentage)
    
    chosen_loss = pp.get_param(pp.PARAM_NAME.LOSS, solution)  
    chosen_optimizer = pp.get_param(pp.PARAM_NAME.OPTIMIZER, solution)  
    learning_rate = pp.get_param(pp.PARAM_NAME.LEARNING_RATE, solution)      



    return self._evaluate(n_devices, m_batch_size, learning_rate, quantization_precission, selected_workers_per_aggregation, local_steps, threshold_percentage, chosen_topology, chosen_loss, chosen_optimizer, is_storing_gradients, speeds, ratio_type, zeros_technique, incremental_max_local_steps, nn_seed, pp.M_BATCH_SIZE_VALUES[0])
     




  #def evaluate(self, solution, chosen_topology, nn_seed = 1):
  def _evaluate(self, n_devices, m_batch_size, learning_rate, quantization_precission, selected_workers_per_aggregation, local_steps, threshold_percentage, chosen_topology, chosen_loss, chosen_optimizer, is_storing_gradients, speeds, ratio_type, zeros_technique, incremental_max_local_steps ,nn_seed, batch_size_worst_case):


    global random
    logging.info("NEW EVALUATION")
    random_state = random.getstate()
    nprandom_state = np.random.get_state()
    random.seed(nn_seed)
    myrandom_seed(nn_seed) 

    #FINALIZATION CRITERIA
    max_epochs = 1
    
    max_global_aggregations = 99999999 #3 #300 #99999999 
    normalized_local_steps = 0 # local steps of the slower node

    Xs_train, ys_train = divide_data_between(self.X_train, self.y_train, n_devices)
    total_m_batches_local = int(Xs_train[0].shape[0] / m_batch_size)
    total_m_batches_local_in_worst_case = int(Xs_train[0].shape[0] / batch_size_worst_case)
    logging.info("Xs_train.shape = " + str(Xs_train[0].shape))
    logging.info("total_m_batches_local = " + str(total_m_batches_local))    

    
    ##TODO -> TERMINATION CRITERION 
    max_local_steps = total_m_batches_local * incremental_max_local_steps
    
    workers = []
    for i in range(len(Xs_train)):
      worker = JSDWorker(Xs_train[i], ys_train[i], chosen_topology, chosen_optimizer, learning_rate, chosen_loss, local_steps, m_batch_size, threshold_percentage, quantization_precission, is_storing_gradients, speeds[i]) #TODO tiempo de ejecución 1 es el tiempo normalizado. 2 es más rápido. 4 más rápido. El nº de local steps se calcula a partir del tiempo de ejecución. El local steps inicial es el que utiliza el dispositivo más lento.
      workers.append(worker)
  
    aggregator = JSDAggregator(Xs_train[i], ys_train[i], chosen_topology, chosen_optimizer, learning_rate, chosen_loss, local_steps, m_batch_size, threshold_percentage, quantization_precission, ratio_type, zeros_technique)


  
    while normalized_local_steps < max_local_steps and aggregator.global_aggregations_counter < max_global_aggregations:
      selected_devices = random.sample(range(n_devices), selected_workers_per_aggregation)  


      
      #parallel
      to_aggregate = []
      #UPDATE LOCAL MODELS -> Send global model to local model (only selected)
      logging.info("selected_devices = " + str(selected_devices)) 
      for device_index in selected_devices:
        logging.info("device_index = " + str(device_index))
        selected_worker = workers[device_index]
 
        to_aggregate.append(selected_worker) #(selected_worker.model.trainable_variables)
 
      #for w in to_aggregate: 
      #  w.assign(aggregator.model.trainable_variables)        
      #  w.train() 
      #print("to_aggregate = " + str(to_aggregate))
      list(map(mytrain, to_aggregate, [aggregator] * len(to_aggregate)))
      aggregator.global_aggregation(to_aggregate) #aggregator.global_aggregation_weights(to_aggregate)      
      #parallel
      

      
      normalized_local_steps += local_steps
      logging.info("end aggregation! normalized_local_steps = " + str(normalized_local_steps) + " global_aggregations = " + str(aggregator.global_aggregations_counter)) 

    communication_fitness = get_comm_fitness(total_m_batches_local_in_worst_case, total_m_batches_local, local_steps, aggregator.model, selected_workers_per_aggregation, n_devices, threshold_percentage, quantization_precission)
    
    acc_fitness = test_model(aggregator.model, self.X_test, self.y_test) 
    acc_fitness = acc_fitness[0]

    random.setstate(random_state)
    np.random.set_state(nprandom_state)

    logging.info("fitness = " + str([acc_fitness, communication_fitness])) 
    return acc_fitness, communication_fitness    
    








