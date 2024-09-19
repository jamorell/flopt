import logging

import math

import tensorflow as tf
import numpy as np
from sklearn.metrics import cohen_kappa_score


def test_model(model, X_test, y_test):
  m = tf.keras.metrics.deserialize({"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}})
  mse = tf.keras.losses.MeanSquaredError() #tf.keras.losses.deserialize({"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}})

  y_pred = model.predict(X_test) 
  loss = mse(y_test, y_pred).numpy()
  m.reset_states()
  m.update_state(y_test, y_pred)
  acc = m.result().numpy()
  #conf_matrix = json.dumps(tf.math.confusion_matrix(labels=tf.argmax(y_test, 1), predictions=tf.argmax(y_pred, 1)).numpy().tolist())
  conf_matrix = tf.math.confusion_matrix(labels=tf.argmax(y_test, 1), predictions=tf.argmax(y_pred, 1)).numpy().tolist()

  ####COHEN KAPPA SCORE
  target_names = range(10)
  true_labels = np.empty(0, "int32")
  predicted = np.empty(0, "int32")

  for i in range(len(conf_matrix)):
    #logging.info(conf_matrix[i])
    temp_true_labels = np.sum(conf_matrix[i]) 
    temp_ok_predicted_labels = conf_matrix[i][i] 
    true_labels = np.append(true_labels, np.full(temp_true_labels, i))
    predicted = np.append(predicted, np.full(temp_ok_predicted_labels, i))
    for k in range(len(conf_matrix[0])):
      if (k != i):
        predicted = np.append(predicted, np.full(conf_matrix[i][k], k))
  ##
  kappa_score_linear = cohen_kappa_score(true_labels, predicted, weights= 'linear')
  kappa_score_quadratic = cohen_kappa_score(true_labels, predicted, weights= 'quadratic')
  ####
  
  return acc, loss, conf_matrix, kappa_score_linear, kappa_score_quadratic




def get_comm_fitness(total_m_batches_local_in_worst_case, total_m_batches_local, local_steps, model, selected_workers_per_aggregation, n_devices, threshold_percentage, quantization_precission):
  #TODO PESOS EN CADA CAPA
  #Calculating communication effort
  total_parameters_in_model = model.count_params()
  percentage_params_per_layer = []
  for i in range(len(model.trainable_variables)):
      percentage_params_per_layer.append(tf.math.reduce_prod(model.trainable_variables[i].shape).numpy() / total_parameters_in_model)
      

  logging.info("len(model.trainable_variables) = " + str(len(model.trainable_variables)))
  logging.info("len(quantization_precission) = " + str(len(quantization_precission)))
  logging.info("len(percentage_params_per_layer) = " + str(len(percentage_params_per_layer)))


  comm_n_devices = selected_workers_per_aggregation / n_devices
  comm_thres_q = 0
  for i in range(len(threshold_percentage)):
      comm_thres_q += (quantization_precission[i] / 32.0) * ((100.0 - threshold_percentage[i]) / 100.0) * percentage_params_per_layer[i]

  t = math.ceil(total_m_batches_local / local_steps) #comm_steps = 1 / local_steps
  t_max = total_m_batches_local_in_worst_case
  t_div_t_max = t / t_max
  communication_fitness = 0.5 * t_div_t_max * comm_n_devices +  0.5 * t_div_t_max * comm_n_devices * comm_thres_q


  logging.info("comm_n_devices = " + str(comm_n_devices))
  logging.info("comm_thres_q = " + str(comm_thres_q))
  logging.info("t_div_t_max = " + str(t_div_t_max))
  logging.info("communication_fitness = " + str(communication_fitness))
  return communication_fitness
  
  



def get_comm_fitness_old(local_steps, model, selected_workers_per_aggregation, n_devices, threshold_percentage, quantization_precission):
  #TODO PESOS EN CADA CAPA
  #Calculating communication effort
  total_parameters_in_model = model.count_params()
  percentage_params_per_layer = []
  for i in range(len(model.trainable_variables)):
      percentage_params_per_layer.append(tf.math.reduce_prod(model.trainable_variables[i].shape).numpy() / total_parameters_in_model)
      

  logging.info("len(model.trainable_variables) = " + str(len(model.trainable_variables)))
  logging.info("len(quantization_precission) = " + str(len(quantization_precission)))
  logging.info("len(percentage_params_per_layer) = " + str(len(percentage_params_per_layer)))


  comm_n_devices = selected_workers_per_aggregation / n_devices
  comm_thres_q = 0
  for i in range(len(threshold_percentage)):
      comm_thres_q += (quantization_precission[i] / 32.0) * ((100.0 - threshold_percentage[i]) / 100.0) * percentage_params_per_layer[i]

  comm_steps = 1 / local_steps 
  communication_fitness = 0.5 * comm_steps * comm_n_devices +  0.5 * comm_steps * comm_n_devices * comm_thres_q


  logging.info("comm_n_devices = " + str(comm_n_devices))
  logging.info("comm_thres_q = " + str(comm_thres_q))
  logging.info("comm_steps = " + str(comm_steps))
  logging.info("communication_fitness = " + str(communication_fitness))
  return communication_fitness

