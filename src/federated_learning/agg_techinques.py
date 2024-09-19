from jsd_utils import *
import tensorflow as tf




    
    


def aggregate_results(aggregator, selected_workers, using_gradients, zeros_technique, ratio_type):
  final_tensors = []  
  arr = []
  for i in range(len(selected_workers)):
    temp =  selected_workers[i].get_gradients() if using_gradients else selected_workers[i].get_weights()    
    arr.append(temp)

  ratio = calculate_agg_ratio(selected_workers, ratio_type)

  for layer in range(len(arr[0])):
    toavg = []
    for client in range(len(arr)):
      toavg.append(arr[client][layer])
    toavg = apply_zeros_techinque(toavg, ratio, zeros_technique)
    #toavg = agg_skipping_zeros(toavg, ratio)
    final_tensors.append(toavg)
    
  for layer in range(len(aggregator.model.trainable_variables)): 
    toassign = tf.math.add(aggregator.model.trainable_variables[layer], final_tensors[layer]) if using_gradients else final_tensors[layer]
    aggregator.model.trainable_variables[layer].assign(toassign)





def calculate_agg_ratio(selected_workers, ratio_type):
  agg_ratio = []
  if ratio_type.upper() == TYPE_AGG_RATIO.NORMAL: #Equal contribution regardless of the number of local steps
    agg_ratio = [1.0 / len(selected_workers)] * len(selected_workers)
  else:
    speeds_sum = 0
    for i in range(len(selected_workers)):
      speeds_sum += selected_workers[i].speed    
    if ratio_type.upper() == TYPE_AGG_RATIO.CORR: #Correlated. More local steps -> more contribution in aggregation
      for i in range(len(selected_workers)):
        agg_ratio.append(selected_workers[i].speed / speeds_sum)
    elif ratio_type.upper() == TYPE_AGG_RATIO.INV_CORR: #Inversely Correlated. More local steps -> less contribution in aggregation
      for i in range(len(selected_workers)):
        agg_ratio.append(1 - (selected_workers[i].speed / speeds_sum))     
    else:
      raise Exception("Aggregation ratio not configured!")   
  return agg_ratio



def apply_zeros_techinque(tensors, ratio, zeros_technique):
  if zeros_technique.upper() == TYPE_ZEROS_TECHNIQUE.IGNORE:
    return agg_ignoring_zeros(tensors, ratio)
  elif zeros_technique.upper() == TYPE_ZEROS_TECHNIQUE.SKIP:
    return agg_skipping_zeros(tensors, ratio)
  elif zeros_technique.upper() == TYPE_ZEROS_TECHNIQUE.FORCE:
    return agg_force_zeros(tensors, ratio)

def agg_ignoring_zeros(tensors, ratio): 
  """
  Average tensors.
  For example if we have two tensors [2, 8, 4] and [0, 2, 2] the result will be [1, 5, 3].
  :param list tensors: List of tensorflow tensors
  :return: tensorflow tensor
  """
  thelength = len(tensors)
  #sum_tensors = tf.math.add_n(tensors)
  #mydiv = tf.math.divide_no_nan(sum_tensors, thelength)
  #toadd = tf.keras.layers.Multiply()([tensors, ratio])
  #mydiv = tf.math.add_n(toadd) 
  mydiv = apply_ratio(tensors, ratio)
  return mydiv  



def agg_skipping_zeros(tensors, ratio): 
  """
  Average tensors skipping zero values.
  If we have 4 tensors and one value is equal to zero 2 times, we will average this value using n = 2.
  For example if we have two tensors [2, 8, 4] and [0, 2, 2] the result will be [2, 5, 3].
  :param list tensors: List of tensorflow tensors
  :return: tensorflow tensor
  """
  temp = tf.multiply(tensors, 1.0)
  total_non_zeros = tf.reduce_sum(tf.cast(temp != 0, tf.float32), 0)

  
  ratio_tensors = []
  ratio_tensors_zero = []
  for client in range(len(tensors)): # each worker tensor
    non_zeros = tf.cast(tensors[client] != 0, tf.float32)
    ratio_tensors.append(tf.math.multiply(non_zeros, ratio[client]))
    
    zeros = tf.cast(tensors[client] == 0, tf.float32)
    ratio_tensors_zero.append(tf.math.multiply(zeros, ratio[client]))
    
  ratio_tensors = tf.multiply(ratio_tensors, 1.0)
  ratio_tensors_zero = tf.multiply(ratio_tensors_zero, 1.0)

  ratio_tensors_zero = tf.math.divide_no_nan(ratio_tensors_zero, total_non_zeros)
  

  ratio_tensors_bool = tf.cast(ratio_tensors != 0, tf.float32)
  ratio_tensors_zero = tf.reduce_sum(ratio_tensors_zero, 0)
  
  ratio_tensors_zero = [ratio_tensors_zero] * len(tensors)
  ratio_tensors_zero = tf.multiply(ratio_tensors_zero, ratio_tensors_bool)
  ratio_tensors = tf.math.add(ratio_tensors, ratio_tensors_zero)
  
  temp = tf.reduce_sum(tf.math.multiply(temp, ratio_tensors), 0)
  return temp  

def agg_force_zeros(tensors, ratio):
  """
  Average tensors forcing zero when a value is zero in one tensor.
  If we have 4 tensors and one value is equal to zero, this value will be zero in the result tensor.
  For example if we have two tensors [2, 8, 4] and [0, 2, 2] the result will be [0, 5, 3].
  :param list tensors: List of tensorflow tensors
  :return: tensorflow tensor
  """
  non_zero = None
  for client in range(len(tensors)): # each worker tensor
    temp_non_zero = tf.cast(tensors[client] != 0, tf.float32)
    if non_zero is None:
      non_zero = temp_non_zero
    else:
      non_zero = tf.multiply(non_zero, temp_non_zero)
      del temp_non_zero
  # Put zeros for average       
  for client in range(len(tensors)): # each worker tensor   
    mytemp = tensors[client]
    tensors[client] = tf.multiply(mytemp, non_zero)
    del mytemp
  
  #return tensors  
  #toadd = tf.keras.layers.Multiply()([tensors, ratio])
  #mydiv = tf.math.add_n(toadd)     
  mydiv = apply_ratio(tensors, ratio)
  return mydiv
  ################    
  
  
def apply_ratio(tensors, ratio):
  for i in range(len(tensors)):
    todelete = tensors[i]
    tensors[i] = tf.math.multiply(tensors[i], ratio[i])
    del todelete
  toreturn = tf.math.add_n(tensors)
  for i in range(len(tensors)):
    todelete = tensors[i]
    del todelete
  del tensors   
  return toreturn
  