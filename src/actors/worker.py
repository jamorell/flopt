import logging

from nn.model import *

from jsd_utils import _get_mypercentile  

def loss(model, x, y, training, loss_object):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=False)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, loss_object):
  with tf.GradientTape() as tape:
    #logging.info("zzz loss_object = " + str(loss_object))
    loss_value = loss(model, inputs, targets, True, loss_object)
    #logging.info("zzz loss_value = " + str(loss_value))
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
  
  
  

class JSDWorker:
  def __init__(self, X_train, y_train, chosen_topology, chosen_optimizer, lr, chosen_loss, local_steps, m_batch_size, threshold_percentage, quantization_precission , is_storing_gradients, speed):
    self.threshold_percentage = threshold_percentage
    self.quantization_precission = quantization_precission
    self.X_train = X_train
    self.y_train = y_train
    self.model = create_topology(chosen_topology)
    self.learning_rate = lr
    self.optimizer = create_optimizer(chosen_optimizer, lr)
    self.loss = create_loss(chosen_loss)
    self.local_steps = local_steps * speed
    self.m_batch_size = m_batch_size # 8
    self.total_m_batches_local = self.X_train.shape[0] / self.m_batch_size
    self.epoch = 0
    self.step_in_epoch = 0
    #self.total_local_steps = 0
    #If storing gradients instead of weights
    self.is_storing_gradients = is_storing_gradients
    self.stored_gradients = None
    self.mypercentile = _get_mypercentile()
    self.speed = speed
    
    # Calculate maximum local steps for finalizing epoch
    self.total_local_steps_performed = 0
    self.maximum_local_steps_per_epoch = int(((self.total_m_batches_local // local_steps) * local_steps * speed) +  ((self.total_m_batches_local % local_steps) * speed))
    
    #DEBUG
    logging.info("....MAX NUMBER OF AGGREGATIONS = " + str((self.total_m_batches_local // local_steps) + (1 if self.total_m_batches_local % local_steps > 0 else 0)))
    logging.info("....m_batch_size = " + str(self.m_batch_size))    
    logging.info("....total_m_batches_local = " + str(self.total_m_batches_local)) 
    logging.info("....local_steps = " + str(local_steps))    
    logging.info("....self.local_steps = " + str(self.local_steps))           
    logging.info("....speed = " + str(self.speed))      
    logging.info("....maximum_local_steps_per_epoch = " + str(self.maximum_local_steps_per_epoch))    
    #TODO-> Random batch
    #TODO-> Shuffle data


    
  def train(self):
    #Local steps
    logging.info("self.step_in_epoch = " + str(self.step_in_epoch) + " training_steps = " + str(self.local_steps))
    counter = 0
    while counter < self.local_steps and self.total_local_steps_performed < self.maximum_local_steps_per_epoch:
      if self.step_in_epoch >= self.total_m_batches_local:  
        self.step_in_epoch = 0
        self.epoch += 1 
      self._train()
      self.step_in_epoch += 1
      counter += 1
      self.total_local_steps_performed += 1
      
  

  def _get_stored_gradients(self):
    toreturn = []
    for i in range(len(self.model.trainable_variables)): 
      #print("self.stored_gradients[i] " + str(self.stored_gradients[i]))
      toreturn.append(tf.math.subtract(self.model.trainable_variables[i], self.stored_gradients[i]))
    del self.stored_gradients
    return toreturn
     
  def get_gradients(self):
    tensors = self._get_stored_gradients() 
    return self._apply_operations(tensors)
      
  def get_weights(self):
    tensors = self.model.trainable_variables
    return self._apply_operations(tensors)
    
  def _apply_operations(self, tensors):
    
 
    logging.info(len(tensors))
    logging.info(len(self.threshold_percentage))
    for i in range(len(tensors)):
      if self.threshold_percentage[i] > 0:
        threshold = self.mypercentile(tf.math.abs(tensors[i]), q=self.threshold_percentage[i]) 
      else:
        threshold = 0
        
      threshold_applied =  tf.where( tf.math.greater_equal(tf.math.abs(tensors[i]), threshold), tensors[i] * 1.0, tensors[i] * 0.0)
      
      quantized = tf.quantization.quantize_and_dequantize(threshold_applied,
                                                            input_min = 0,
                                                            input_max= 1,
                                                            num_bits = self.quantization_precission[i],
                                                            signed_input = True,
                                                            range_given=False)
      todelete = tensors[i]
      tensors[i] = quantized
      del todelete
    return tensors     
    
    
  def _train(self):
    #Last batch can be of different length
    first = int(self.step_in_epoch * self.m_batch_size)
    last = int(first + self.m_batch_size)
    local_x = self.X_train[first:last]
    local_y = self.y_train[first:last]
    #####################if local_x.shape[0] == 0: # TODO -> Revisar esto
    #if local_x.shape[0] == 0: # TODO -> Revisar esto
    #  logging.error("local_x.shape[0] == 0 SKIPPING _TRAIN ")
    #  return False
    loss_value, grads = grad(self.model, local_x, local_y, self.loss)
    


    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
 

    
  def assign(self, trainable_variables):
    if self.is_storing_gradients:
      self.stored_gradients = []
      for i in range(len(self.model.trainable_variables)):
        self.stored_gradients.append(tf.identity(self.model.trainable_variables[i])) 
      
      
    #logging.info("assign trainable_variables = " + str(trainable_variables))
    for i in range(len(self.model.trainable_variables)):
      #logging.info("assigning i = " + str(i))
      self.model.trainable_variables[i].assign(trainable_variables[i])       
      #logging.info("end assign")
      
      
