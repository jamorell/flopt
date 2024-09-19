#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#https://stackoverflow.com/questions/53732589/how-to-set-upper-and-lower-bounds-to-a-gene-in-an-individual-in-deap

#learning_rate
#m_batch_size
#weights_or_gradients
#ratio_type
#zeros_technique  

import sys
import array
import random
import json


import numpy as np


import os
import time


start_time = time.time()

from math import sqrt

    
import matplotlib.pyplot as plt

import argparse


from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

import logging
logging.basicConfig(
    #filename='HISTORYlistener.log',
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)



from FL_OF import FL_OF
from fl import JSDEvaluate 

from jsd_utils import *

from operators.my_strategy import *



class ProblemParameters():
#learning_rate
#m_batch_size
#weights_or_gradients
#ratio_type
#zeros_technique   
  PARAM_NAME = Enum(["QUANTIZATION", "SELECTED_DEVICES", "LOCAL_STEPS", "SPARSIFICATION"
  , "LEARNING_RATE", "M_BATCH_SIZE", "WEIGHTS_OR_GRADIENTS", "RATIO", "ZEROS_TECHNIQUE"
  , "OPTIMIZER", "LOSS", "INCREMENTAL_MAX_LOCAL_STEPS"])
  INIT_TYPE = Enum(["RANDOM", "LOWER", "UPPER", "MAX_COMM"])
  
  MUTATION_TYPE = Enum(["mutUniformInt", "myMutInt"])
  DATASET_TYPE = Enum(["MNIST", "FASHION"])
  
  STRATEGY_TYPE = Enum(["NONE", "EDA", "CMAES"])

  LOSS_DEFAULT = TYPE_LOSS.CategoricalCrossentropy

  OPTIMIZER_CONV_DEFAULT = TYPE_OPTIMIZER.SGD
  LEARNING_RATE_CONV_DEFAULT = 0.01
  LEARNING_RATE_CONV_VALUES = [0.005, LEARNING_RATE_CONV_DEFAULT, 0.015, 0.02]

  OPTIMIZER_DENSE_DEFAULT = TYPE_OPTIMIZER.Adam  
  LEARNING_RATE_DENSE_DEFAULT = 0.001
  LEARNING_RATE_DENSE_VALUES = [0.0005, LEARNING_RATE_DENSE_DEFAULT, 0.0015, 0.002]
  
  M_BATCH_SIZE_DEFAULT = 8
  M_BATCH_SIZE_VALUES = [4, M_BATCH_SIZE_DEFAULT, 16, 32]
  
  WEIGHTS_OR_GRADIENTS_DEFAULT = TYPE_WEIGHTS_OR_GRADIENTS.GRADIENTS
  WEIGHTS_OR_GRADIENTS_VALUES = [TYPE_WEIGHTS_OR_GRADIENTS.WEIGHTS, TYPE_WEIGHTS_OR_GRADIENTS.GRADIENTS]
  
  RATIO_DEFAULT = TYPE_AGG_RATIO.NORMAL
  RATIO_VALUES = [TYPE_AGG_RATIO.NORMAL, TYPE_AGG_RATIO.CORR, TYPE_AGG_RATIO.INV_CORR]
  
  ZEROS_TECHNIQUE_DEFAULT = TYPE_ZEROS_TECHNIQUE.SKIP
  ZEROS_TECHNIQUE_VALUES = [TYPE_ZEROS_TECHNIQUE.IGNORE, TYPE_ZEROS_TECHNIQUE.SKIP, TYPE_ZEROS_TECHNIQUE.FORCE]



  
  def __init__(self,  chosen_topology, n_layers, n_devices, speeds, ga_seed):
    self.strategy_type = self.STRATEGY_TYPE.EDA  
    self.n_layers = n_layers
    self.n_devices = n_devices    
    self.speeds = speeds
    self.ga_seed = ga_seed

    self.chosen_topology = chosen_topology #"DENSE" #"CONV"
    self.N_ITERATIONS = 52 #302 #3000
    self.POP_SIZE = 4 #100
    self.CXPB = 0.9
    self.params = {}
    self._size = 0    
    
    self.mutation = self.MUTATION_TYPE.mutUniformInt    
    self._initial_incremental_max_local_steps = 1.0 #0.4 #0.05 #1.0
    self._incremental_max_local_steps = self._initial_incremental_max_local_steps
    self._incremental_amount = 0.2 #0.05
    self._incremental_after_n_generations = 5    
    
    self.hypervolumes = []
    #probabilities of each value in each gene
    self.prob_gen = []
    self.prob_rep = [] 
    self.prob_decrement = [] 
    self.lower_bounds = []
    self.upper_bounds = []  
    

  def reset(self):    
    self.hypervolumes = []

    self.prob_gen = self.prob_gen_cp.copy()
    self.prob_rep = self.prob_rep_cp.copy()
    self.prob_decrement = self.prob_decrement_cp .copy()   

  def update_incremental_max_local_steps(self, num_gen):
    entire_pop_need_to_be_reevaluated = False
    if self._incremental_max_local_steps >= 1.0:
      return entire_pop_need_to_be_reevaluated
    
    if num_gen > 0 and num_gen % self._incremental_after_n_generations == 0:
      self._incremental_max_local_steps += self._incremental_amount
      entire_pop_need_to_be_reevaluated = True
    if self._incremental_max_local_steps > 1.0:
      self._incremental_max_local_steps = 1.0
      
    return entire_pop_need_to_be_reevaluated
    
  def compile(self):
    self.MUTATEPB = 1.0 / self._size #2.0 / self._size
    
    self.prob_gen_cp = self.prob_gen.copy()
    self.prob_rep_cp = self.prob_rep.copy()
    self.prob_decrement_cp = self.prob_decrement.copy()
    

  def add_parameter(self, name, lower_bound, upper_bound, repeat = 1):
    self.params[name] = [self._size, lower_bound, upper_bound, repeat] #init_pos, lower_bound, upper_bound, repeat   
    self._size += repeat
    #probabilities of each value in each gene
    for i in range(repeat):
      ngen = upper_bound - lower_bound + 1
      p = [1.0/(ngen)] * ngen
      self.prob_gen.append(p) 
      self.prob_rep.append([0] * ngen)
      self.prob_decrement.append([1] * ngen)
      self.lower_bounds.append(lower_bound)
      self.upper_bounds.append(upper_bound)
        
  def update_probs(self, probs, decrement):
    original_prob = 1.0 /  len(probs)
    total_values = sum(decrement)
    for i in range(len(probs)):
      probs[i] = (1.0 / (total_values)) * decrement[i] 
      
    #print(probs)
    
    #v = np.random.choice(np.arange(0, len(probs)), p=probs)
    #print(v)    


        
  def  count_pop(self, pop):

    for pos in range(len(pop[0])):
      
      lower = self.lower_bounds[pos]
      upper = self.upper_bounds[pos]
      npositions = upper - lower + 1
      
      counter = [0] * npositions
      for i in range(len(pop)):
        value = round(pop[i][pos]) - lower
        counter[value] +=1
      #print(counter)
      #print(self.prob_rep[pos])
      
      for i in range(len(counter)):
        if counter[i] == 0 and self.prob_rep[pos][i] > 0:
          print("Gene value has disappeared!")
          self.prob_decrement[pos][i] /= pp.decr #2
        self.prob_rep[pos][i] = counter[i]
      #print("printing")
      #print(self.prob_decrement[pos])
      #print(self.prob_gen[pos])
      self.update_probs(self.prob_gen[pos], self.prob_decrement[pos])
    print("\n\n\n")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! self.prob_rep = " + str(self.prob_rep))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! self.prob_decrement = " + str(self.prob_decrement))    
    
      #print(self.prob_gen[pos])
      #print(sum(self.prob_gen[pos]))
      #print(self.prob_rep)

     


    
  def init_solution(self, init_type):
    solution = [0] * self._size
    
    for key in self.params:
      p = self.params[key]
      init_pos = p[0]
      lower_bound = p[1]
      upper_bound = p[2]
      repeat = p[3]
      for i in range(init_pos, init_pos+repeat):
        if init_type.upper() == self.INIT_TYPE.RANDOM:
          solution[i] = random.randint(lower_bound, upper_bound)
        elif init_type.upper() == self.INIT_TYPE.LOWER:      
          solution[i] = lower_bound
        elif init_type.upper() == self.INIT_TYPE.UPPER:      
          solution[i] = upper_bound          
        elif init_type.upper() == self.INIT_TYPE.MAX_COMM:
          # Initialize with max communication. Max number of global aggregations
          if key == self.PARAM_NAME.QUANTIZATION:
            solution[i] = upper_bound
          elif key == self.PARAM_NAME.SPARSIFICATION:
            solution[i] = lower_bound    
          elif key == self.PARAM_NAME.SELECTED_DEVICES:
            solution[i] = upper_bound   
          elif key == self.PARAM_NAME.LOCAL_STEPS:
            solution[i] = lower_bound   
          elif key == self.PARAM_NAME.LEARNING_RATE:
            solution[i] = 3#0#1 # equal to DEFAULT
          elif key == self.PARAM_NAME.M_BATCH_SIZE:
            solution[i] = 1#1#0#1 #0 #1 # equal to DEFAULT     
          elif key == self.PARAM_NAME.WEIGHTS_OR_GRADIENTS:
            solution[i] = 0#1 # equal to DEFAULT       
          elif key == self.PARAM_NAME.RATIO:
            solution[i] = 0 # equal to DEFAULT              
          elif key == self.PARAM_NAME.ZEROS_TECHNIQUE:
            solution[i] = 0#1 # equal to DEFAULT                            
          else:
            raise Exception("Parameter name not configured! " + key)  
            
            
    #if init_type.upper() == self.INIT_TYPE.MAX_COMM:
    #  print(init_type.upper())
    #  print(solution)

    return solution


  def get_param(self, name, solution):   
    take_from = None
    if name == self.PARAM_NAME.LEARNING_RATE:    
      if self.chosen_topology == TYPE_TOPOLOGY.DENSE:
        #if name not in solution:
        #  return self.LEARNING_RATE_DENSE_DEFAULT 
        #else: 
        take_from = self.LEARNING_RATE_DENSE_VALUES      
      elif self.chosen_topology == TYPE_TOPOLOGY.CONV:
        #if name not in solution:
        #  return self.LEARNING_RATE_CONV_DEFAULT 
        #else: 
        take_from = self.LEARNING_RATE_CONV_VALUES    
      else: 
        raise Exception("Topology name not configured! " + self.chosen_topology)      
    elif name == self.PARAM_NAME.M_BATCH_SIZE:
      #if name not in solution:
      #  return self.M_BATCH_SIZE_DEFAULT 
      #else: 
      take_from = self.M_BATCH_SIZE_VALUES
    elif name == self.PARAM_NAME.WEIGHTS_OR_GRADIENTS:
      #if name not in solution:
      #  return self.WEIGHTS_OR_GRADIENTS_DEFAULT 
      #else: 
      take_from = self.WEIGHTS_OR_GRADIENTS_VALUES
    elif name == self.PARAM_NAME.RATIO:
      #if name not in solution:
      #  return self.RATIO_DEFAULT 
      #else: 
      take_from = self.RATIO_VALUES   
    elif name == self.PARAM_NAME.ZEROS_TECHNIQUE:
      #if name not in solution:
      #  return self.ZEROS_TECHNIQUE_DEFAULT 
      #else: 
      take_from = self.ZEROS_TECHNIQUE_VALUES      
      
    elif name == self.PARAM_NAME.OPTIMIZER:    
      if self.chosen_topology == TYPE_TOPOLOGY.DENSE:
        return self.OPTIMIZER_DENSE_DEFAULT
      elif self.chosen_topology == TYPE_TOPOLOGY.CONV:
        return self.OPTIMIZER_CONV_DEFAULT
      else: 
        raise Exception("Topology name not configured! " + self.chosen_topology)       
    elif name == self.PARAM_NAME.LOSS:    
      return self.LOSS_DEFAULT 
    elif name == self.PARAM_NAME.INCREMENTAL_MAX_LOCAL_STEPS:
      return self._incremental_max_local_steps     

        
    p = self.params[name]
    init_pos = p[0]
    lower_bound = p[1]
    upper_bound = p[2]
    repeat = p[3]

    if repeat > 1:    
      arr = []
      idx = init_pos    
      while idx < init_pos + repeat:
        arr.append(solution[idx])
        idx += 1  
      return arr
    else:
      if take_from is None:
        return solution[init_pos] 
      else:  
        return take_from[solution[init_pos]]
    
  

def myinitialization_with_max_comm(pp):
    return pp.init_solution(pp.INIT_TYPE.MAX_COMM)

def myinitialization(pp):
    return pp.init_solution(pp.INIT_TYPE.RANDOM)





from scoop import futures
import multiprocessing #for counting number of cores







toolbox = base.Toolbox()
toolbox.register("map", futures.map)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)



#####
def myMutInt(individual, pp):
    for pos in range(pp._size):
      if random.random() < pp.MUTATEPB:
        lower = pp.lower_bounds[pos]
        upper = pp.upper_bounds[pos]
        npositions = upper - lower + 1     
        probs = pp.prob_gen[pos]
        print("MUTATE probs = " + str(probs))
        print("MUTATE sum probs = " + str(sum(probs)))        
        v = np.random.choice(np.arange(0, len(probs)), p=probs)
        v += lower
        individual[pos] = v


    return individual,
    
    
def myCXUniform(ind1, ind2, pp):    
  selected = np.random.uniform(low=0.0, high=1.0, size=pp._size)
  p1_selected = np.where(selected < 0.5, 1, 0)
  p2_selected = 1 - p1_selected
  offspring1 = np.add(np.multiply(ind1, p1_selected), np.multiply(ind2, p2_selected))
  offspring2 = np.add(np.multiply(ind2, p1_selected), np.multiply(ind1, p2_selected))  
  return (offspring1, offspring2)
  
####


def main(pp):
    global toolbox
    random.seed(pp.ga_seed)
    np.random.seed(pp.ga_seed) 

    
    ls_strategy = None
    if pp.strategy_type != pp.STRATEGY_TYPE.NONE:
      lambda_ = 10 # n best accuracies for distribution
      mu = int(lambda_ / 2.0)
      if pp.strategy_type == pp.STRATEGY_TYPE.EDA:
        ls_strategy = EDA_Strategy(centroid=[0]*pp._size, sigma=[1.0]*pp._size, lambda_=lambda_, mu = mu)
      #elif pp.strategy_type == pp.STRATEGY_TYPE.CMAES:
      #  ls_strategy = CMAES_Strategy(centroid=[0]*pp._size, sigma=0.1, lambda_=lambda_)
      #ls_strategy = CMAES_Strategy(centroid=[0]*NDIM, sigma=0.1, lambda_=lambda_)    
      #ls_strategy = CMAES_Strategy(centroid=[0]*pp._size, sigma=1.0, lambda_=10*1)
    
    ev = JSDEvaluate(pp)



    toolbox.register("attr_int", myinitialization, pp)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("evaluate", FL_OF, evaluator = ev, problem_params = pp)

    #toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low = lower_bounds, up = upper_bounds)
    #toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mate", myCXUniform, pp=pp)
    
    
    #toolbox.register("mutate", tools.mutUniformInt, low = lower_bounds, up = upper_bounds, indpb=MUTATEPB)
    #toolbox.register("mutate", tools.mutPolynomialBounded, low=lower_bounds, up=upper_bounds, eta=20.0, indpb=pp.MUTATEPB)
    #toolbox.register("mutate", tools.mutUniformInt, low=pp.init_solution(pp.INIT_TYPE.LOWER), up=pp.init_solution(pp.INIT_TYPE.UPPER), indpb=pp.MUTATEPB)
    toolbox.register("mutate", myMutInt, pp=pp)
    
    if pp.mutation.upper() == pp.MUTATION_TYPE.mutUniformInt.upper():
      toolbox.register("mutate", tools.mutUniformInt, low=pp.init_solution(pp.INIT_TYPE.LOWER), up=pp.init_solution(pp.INIT_TYPE.UPPER), indpb=pp.MUTATEPB)
    elif pp.mutation.upper() == pp.MUTATION_TYPE.myMutInt.upper():
      toolbox.register("mutate", myMutInt, pp=pp) 
    
    toolbox.register("select", tools.selNSGA2)

    ######################################


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n = pp.POP_SIZE)
    #TODO -> counting parameters in pop
    pp.count_pop(pop)
    #
    
    if pp.start_with_max_comm: #Adding one individual with max communication
      pop[0] = creator.Individual(myinitialization_with_max_comm(pp)) # add default individual
    

    # Evaluate the individuals with an invalid fitness
    logging.info("#### FIRST EVALUATION ")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    logging.info(logbook.stream)

    # Begin the generational process
    for gen in range(1, pp.N_ITERATIONS):
        print("\n")
        logging.info("#### ITERATION = " + str(gen))
        entire_pop_need_to_be_reevaluated = pp.update_incremental_max_local_steps(gen)
        
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= pp.CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
         
            del ind1.fitness.values, ind2.fitness.values
        
        #CMA-ES
        if pp.strategy_type != pp.STRATEGY_TYPE.NONE:
            ls_strategy.update(pop, pp.init_solution(pp.INIT_TYPE.LOWER), pp.init_solution(pp.INIT_TYPE.UPPER))
            ls_offspring = ls_strategy.generate(creator.Individual, pp.init_solution(pp.INIT_TYPE.LOWER), pp.init_solution(pp.INIT_TYPE.UPPER))
            print(offspring)
            print(ls_offspring)            
            '''
            ls_offspring = np.clip(ls_offspring, pp.init_solution(pp.INIT_TYPE.LOWER), pp.init_solution(pp.INIT_TYPE.UPPER)) #Checking bounds max and mix value in each position
            '''
            offspring += ls_offspring
            print(offspring)
            #exit()
        #
        invalid_ind = None
        if pp.repeat_evaluate_pop and entire_pop_need_to_be_reevaluated:
          invalid_ind = pop + offspring
        else:        
          # Evaluate the individuals with an invalid fitness
          invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
          
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, pp.POP_SIZE)
        record = stats.compile(pop)
        logbook.record(gen = gen, evals = len(invalid_ind), **record)
        logging.info(logbook.stream)

        #TODO -> counting parameters in pop
        pp.count_pop(pop)
        #


        save_stats(pop, logbook, gen, pp)

    #logging.info("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook
        
if __name__ == "__main__":
    #with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #    optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    #optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", required=True, help="Seed for optimizer algorithm.", type=int)
    ap.add_argument("-i", "--iterations", required=True, help="Number of iterations.", type=int)
    ap.add_argument("-p", "--population", required=False, help="Population of the optimization algorithm.", type=int, default=100)            
    ap.add_argument("-t", "--topology", required=True, help="NN model topology. DENSE or CONV.")
    ap.add_argument("-l", "--layers", required=True, help="Number of layers for NN model.", type=int)
    ap.add_argument("-d", "--devices", required=True, help="Number of devices.", type=int, default=4)
    ap.add_argument("-sp", "--speeds", required=True, help="Speeds of devices.", default="1.0,1.0,1.0,1.0")
    ap.add_argument("-maxc", "--max_comm", required=False, help="Start with one individual using max communication.", type=int, default=1)    
    ap.add_argument("-r", "--repeat_evaluate_pop", required=False, help="When we increase the total number of steps we reevaluate the entire population.", type=int, default=1)        
    ap.add_argument("-ii", "--incremental_initial", required=True, help="Initial value of total number of steps for evaluation of NN learning.", type=float)
    ap.add_argument("-ia", "--incremental_amount", required=True, help="The amount we increase the total number of steps for evaluation of NN learning.", type=float)    
    ap.add_argument("-ig", "--incremental_gens", required=True, help="After n generations we increase the total number of steps for evaluation of NN learning.", type=int)  
      
    ap.add_argument("-m", "--mutation", required=True, help="Select mutation operator between [mutUniformInt, myMutInt]")  
    ap.add_argument("-dt", "--dataset", required=True, help="Select mutation operator between [MNIST, FASHION]")  
    ap.add_argument("-decr", "--decrement", required=True, help="When using the extinct value discriminating mutation operator (myMutInt). The amunt to decrement the probability.", type=int)      
    ap.add_argument("-st", "--strategy", required=True, help="Strategy used for top 10 accuracy-fitness individuals in population [NONE, EDA, CMAES]")          
    args = ap.parse_args()
    argparse_dict = vars(args)

    print(argparse_dict)



    
    #speeds = [1.0, 1.5, 3.0, 3.0]
    #n_devices = 4
    ga_seed = int(args.seed)
    chosen_topology = args.topology
    nlayers = int(args.layers)
    n_devices = int(args.devices)
    speeds = np.array(args.speeds.split(',')).astype(np.float)

    print(speeds)

    print(argparse_dict)

    print("ga_seed = " + str(ga_seed))
    print("CHOSEN_TOPOLOGY = " + str(chosen_topology))
    print("nlayers = " + str(nlayers))
    myseeds = [62011,80177,97213,109567,181327103,117797,122393,130841,137803,141223,144961,149749,155657,159193,163679,167801,173137,184649,189407,198529,204047,208843,214789,221077,227219,233297,200604289,251623,256423,263387, 179426549, 1300609]
    ga_seed = myseeds[ga_seed]

    pp = ProblemParameters(chosen_topology, nlayers, n_devices, speeds, ga_seed) ##pp = problem_params
    pp.argparse_dict = argparse_dict
    pp.start_time = start_time
    pp.N_ITERATIONS = int(args.iterations)
    pp.POP_SIZE = int(args.population)
    pp.repeat_evaluate_pop = True if args.repeat_evaluate_pop == int(args.repeat_evaluate_pop) else False
    pp.start_with_max_comm = True if args.max_comm == int(args.max_comm) else False
    #Mutation operator
    if args.mutation.upper() == pp.MUTATION_TYPE.mutUniformInt.upper():
      pp.mutation = pp.MUTATION_TYPE.mutUniformInt
    elif args.mutation.upper() == pp.MUTATION_TYPE.myMutInt.upper():
      pp.mutation = pp.MUTATION_TYPE.myMutInt
    
    pp.decr = int(args.decrement)
    #
    
    #STRATEGY
    if args.strategy.upper() == pp.STRATEGY_TYPE.NONE.upper():
      pp.strategy_type = pp.STRATEGY_TYPE.NONE
    elif args.strategy.upper() == pp.STRATEGY_TYPE.EDA.upper():
      pp.strategy_type = pp.STRATEGY_TYPE.EDA    
    elif args.strategy.upper() == pp.STRATEGY_TYPE.CMAES.upper():
      pp.strategy_type = pp.STRATEGY_TYPE.CMAES  
    else:
      raise Exception("Strategy for top individuals not defined! ")  
          
    
    if args.dataset.upper() == pp.DATASET_TYPE.MNIST.upper():
      pp.dataset = pp.DATASET_TYPE.MNIST
    elif args.dataset.upper() == pp.DATASET_TYPE.FASHION.upper():
      pp.dataset = pp.DATASET_TYPE.FASHION
    else:
      raise Exception("Dataset is not defined! " + str(args.dataset))      
    pp._initial_incremental_max_local_steps = args.incremental_initial
    pp._incremental_max_local_steps = pp._initial_incremental_max_local_steps
    pp._incremental_amount = args.incremental_amount
    pp._incremental_after_n_generations = args.incremental_gens
    
    pp.add_parameter(pp.PARAM_NAME.QUANTIZATION, 1, 32, nlayers)
    pp.add_parameter(pp.PARAM_NAME.SELECTED_DEVICES, 1, pp.n_devices, 1)
    pp.add_parameter(pp.PARAM_NAME.LOCAL_STEPS, 1, 1000, 1)
    pp.add_parameter(pp.PARAM_NAME.SPARSIFICATION, 0, 50, nlayers)
    #new parameters
    pp.add_parameter(pp.PARAM_NAME.LEARNING_RATE, 0, 3, 1)
    pp.add_parameter(pp.PARAM_NAME.M_BATCH_SIZE, 0, 3, 1)
    pp.add_parameter(pp.PARAM_NAME.WEIGHTS_OR_GRADIENTS, 0, 1, 1)    
    pp.add_parameter(pp.PARAM_NAME.RATIO, 0, 2, 1)      
    pp.add_parameter(pp.PARAM_NAME.ZEROS_TECHNIQUE, 0, 2, 1)          
    #pp.m_batch_size = 8
    pp.compile()

    path_folder = get_path_folder(pp)
    from pathlib import Path
    testing = Path(path_folder)
    if testing.exists():
      logging.critical("Exit because we already have the stats of this job.")
      exit(0)
    else:
      os.makedirs(path_folder, exist_ok=True)

  
    #for i in range(1,31):
    first_seed = int(args.seed)
    for i in range(first_seed, first_seed + 1):
      pp.ga_seed = myseeds[i]
      pop, stats = main(pp)
      save_stats(pop, stats, pp.N_ITERATIONS, pp)
      pp.reset()  
    


