import logging

import tensorflow as tf

from actors.worker import JSDWorker


#from operations import JSDOperations
    
from jsd_utils import _get_mypercentile  

from federated_learning.agg_techinques import *
    
class JSDAggregator(JSDWorker):

  def __init__(self, X_train, y_train, chosen_topology, chosen_optimizer, lr, chosen_loss, local_steps, m_batch_size, threshold_percentage, quantization_precission, ratio_type, zeros_technique):
    super().__init__(X_train, y_train, chosen_topology, chosen_optimizer, 1.0, chosen_loss, local_steps, m_batch_size, threshold_percentage, quantization_precission, is_storing_gradients = False, speed = 1.0)

    self.global_aggregations_counter = 0
    self.mypercentile = _get_mypercentile()
    self.ratio_type = ratio_type
    self.zeros_technique = zeros_technique
    
  def global_aggregation(self, selected_workers):
    logging.info("Global Aggregation = " + str(self.global_aggregations_counter))  
    using_gradients = selected_workers[0].is_storing_gradients
    aggregate_results(self, selected_workers, using_gradients, self.zeros_technique, self.ratio_type)
    logging.info("END global Aggregation = " + str(self.global_aggregations_counter))  
    self.global_aggregations_counter += 1
       

