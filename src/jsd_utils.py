import os
import sys
import array
import random
import json
import time
import matplotlib.pyplot as plt
import collections
import logging

import multiprocessing #for counting number of cores

import tensorflow as tf

import numpy as np

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


TYPE_LOSS = Enum(["CategoricalCrossentropy"])
TYPE_OPTIMIZER = Enum(["SGD", "Adam"])
TYPE_TOPOLOGY = Enum(["DENSE", "CONV"])
TYPE_WEIGHTS_OR_GRADIENTS = Enum(["WEIGHTS", "GRADIENTS"])
TYPE_AGG_RATIO = Enum(["NORMAL", "CORR", "INV_CORR"])
TYPE_ZEROS_TECHNIQUE = Enum(["IGNORE", "SKIP", "FORCE"])



def _get_myrandom():
  #return tf.compat.v1.random.set_random_seed
  #'''
  if tf.test.gpu_device_name():
    try:
      return tf.compat.v1.random.set_random_seed
    except:
      return tf.random.set_seed
  else:
    return tf.random.set_seed
  #'''  

def _get_mypercentile():
  #return tf.contrib.distributions.percentile
  #'''
  mypercentile = None
  if tf.test.gpu_device_name():
    try:
      mypercentile = tf.contrib.distributions.percentile
    except:
      import tensorflow_probability as tfp
      mypercentile = tfp.stats.percentile
  else:
    import tensorflow_probability as tfp
    mypercentile = tfp.stats.percentile
  
  return mypercentile
  #'''




def get_path_folder(pp):
  name = "./stats_"
  name += pp.dataset
  name += "_"  
  name += pp.chosen_topology
  name += "_"  
  name += str(pp.ga_seed)
  name += "_"
  name += str(pp.strategy_type)
  name += "_"  
  name += str(pp.n_devices)
  name += "_"
  name += str(pp.speeds)
  name += "_"  
  name += str(pp.N_ITERATIONS)  
  name += "_"  
  name += str(pp._initial_incremental_max_local_steps)  
  name += "_"  
  name += str(pp._incremental_amount)  
  name += "_"  
  name += str(pp._incremental_after_n_generations)       
  name += "_"
  name += str(pp.POP_SIZE)
  name += "_"
  name += str(pp.CXPB)
  name += "_"
  name += str(pp.mutation)
  name += "_"  
  name += str(pp.MUTATEPB)
  name += "_"    
  name += str(pp.decr)
  name += "_"                              
  for key in pp.params:
    p = pp.params[key]
    init_pos = p[0]
    lower_bound = p[1]
    upper_bound = p[2]
    repeat = p[3]  
    name += str(lower_bound)
    name += ":"
    name += str(upper_bound)
    name += "_"
  return name

def get_figure_path(path_folder, pp, num_gen):
  name = path_folder +  "/pareto_initt_"
  name += pp.dataset
  name += "_"  
  name += pp.chosen_topology
  name += "_"  
  name += str(pp.ga_seed)
  name += "_"
  name += str(pp.strategy_type)  
  name += "_"
  name += str(pp.n_devices)
  name += "_"
  name += str(pp.speeds)
  name += "_"
  name += str(num_gen) 
  name += "_"  
  name += str(pp.N_ITERATIONS)
  name += "_"    
  name += str(pp._initial_incremental_max_local_steps)  
  name += "_"  
  name += str(pp._incremental_max_local_steps)  
  name += "_"   
  name += str(pp._incremental_amount)  
  name += "_"  
  name += str(pp._incremental_after_n_generations)    
  name += "_"  
  name += str(pp.POP_SIZE)
  name += "_"
  name += str(pp.CXPB)
  name += "_"
  name += str(pp.mutation)
  name += "_"  
  name += str(pp.MUTATEPB)
  name += "_" 
  name += str(pp.decr)
  name += "_"                                 
  for key in pp.params:
    p = pp.params[key]
    init_pos = p[0]
    lower_bound = p[1]
    upper_bound = p[2]
    repeat = p[3]  
    name += str(lower_bound)
    name += "_"
    name += str(upper_bound)
  name += ".pdf"
  return name
  
  
def get_stats_path(path_folder, pp, num_gen):
  name = path_folder +  "/stats_initt_"
  name += pp.dataset
  name += "_"  
  name += pp.chosen_topology
  name += "_"  
  name += str(pp.ga_seed)
  name += "_"
  name += str(pp.strategy_type)  
  name += "_"
  name += str(pp.n_devices)
  name += "_"
  name += str(pp.speeds)
  name += "_"  
  name += str(num_gen) 
  name += "_"  
  name += str(pp.N_ITERATIONS)  
  name += "_"  
  name += str(pp._initial_incremental_max_local_steps)  
  name += "_"  
  name += str(pp._incremental_max_local_steps)  
  name += "_" 
  name += str(pp._incremental_amount)  
  name += "_"  
  name += str(pp._incremental_after_n_generations)    
  name += "_"
  name += str(pp.POP_SIZE)
  name += "_"
  name += str(pp.CXPB)
  name += "_"
  name += str(pp.mutation)
  name += "_"         
  name += str(pp.MUTATEPB)
  name += "_" 
  name += str(pp.decr)
  name += "_"                                 
  for key in pp.params:
    p = pp.params[key]
    init_pos = p[0]
    lower_bound = p[1]
    upper_bound = p[2]
    repeat = p[3]  
    name += str(lower_bound)
    name += "_"
    name += str(upper_bound)
  name += ".json"
  return name
  
  
def tree():
    ''' 
        Recursive dictionnary with defaultdict 
    '''
    return collections.defaultdict(tree)  
  
def save_stats(pop, stats, num_gen, pp):
    pop.sort(key=lambda x: x.fitness.values)
    
    logging.info(stats)
    #logging.info("Convergence: ", convergence(pop, optimal_front))
    #logging.info("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    '''
    front = np.array([ind.fitness.values for ind in pop])
    optimal_front = np.array(optimal_front)
    plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()
    '''


    #path_folder = './stats_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}'.format(pp.ga_seed, pp.N_ITERATIONS, pp.chosen_topology, pp.POP_SIZE, pp.CXPB, pp.MUTATEPB, pp.lb1, pp.ub1, pp.lay, pp.lb2, pp.ub2, pp.esc, pp.lb3, pp.ub3, pp.ts, pp.lb4, pp.ub4, pp.thre)
    path_folder = get_path_folder(pp)
    os.makedirs(path_folder, exist_ok=True)

    # Plot 
    plt.title("Pareto front")
    plt.xlabel("Accuracy")
    plt.ylabel("Communications")
    #plt.plot(front[:,0],front[:,1],"r--") #plt.scatter(front[:,0], front[:,1], c="b")
    front = np.array([ind.fitness.values for ind in pop])  
    plt.scatter(front[:,0], front[:,1], c="b")  
    plt.grid(True)
    plt.axis("tight")
    #plt.show()
    figure_path = get_figure_path(path_folder, pp, num_gen)
    plt.savefig(figure_path, dpi=600)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    #plt.savefig(path_folder + '/pareto_initt_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.pdf'.format(pp.ga_seed, num_gen, pp.N_ITERATIONS, pp.chosen_topology, pp.POP_SIZE, pp.CXPB, pp.MUTATEPB, pp.lb1, pp.ub1, pp.lay, pp.lb2, pp.ub2, pp.esc, pp.lb3, pp.ub3, pp.ts, pp.lb4, pp.ub4, pp.thre), dpi=600)

    solutions=tree()
    # Evaluate the running time
    gen, evals, std, minn, avg, maxx = stats.select("gen", "evals", "std", "min", "avg", "max")
    t=time.time() - pp.start_time

    solutions['population'] = str(pop)
    pop_fitness = []
    for i in range(len(pop)):
        pop_fitness.append(pop[i].fitness.values)
    solutions['population_fitness'] = pop_fitness #str()
    #solutions['convergence'] = convergence(pop, optimal_front)
    #solutions['diversity'] = diversity(pop, optimal_front[0], optimal_front[-1])
    solutions['time']=t
    solutions["parameters"] = pp.argparse_dict 
    ncores = multiprocessing.cpu_count()
    solutions['cores']= multiprocessing.cpu_count() #1#int(os.environ["SLURM_NTASKS"])

    pp.hypervolumes.append(hypervolume(pop, [0, 1.0]))
    solutions['hypervolume'] = pp.hypervolumes 
    solutions['gen'] = gen
    solutions['evals'] = evals

    print(type(std))
    print(type(std[0]))    

    tosave = []
    for i in range(len(avg)):
      tosave.append(avg[i].tolist())
    solutions['avg'] = tosave    
    
    tosave = []
    for i in range(len(std)):
      tosave.append(std[i].tolist())
    solutions['std'] = tosave        
    
    tosave = []
    for i in range(len(minn)):
      tosave.append(minn[i].tolist())
    solutions['min'] = tosave 
    
    tosave = []
    for i in range(len(maxx)):
      tosave.append(maxx[i].tolist())
    solutions['max'] = tosave       
    
    
       

    '''
    solutions['std'] = std
    solutions['min'] = minn
    solutions['avg'] = avg
    solutions['max'] = maxx

    '''    
    random_state = random.getstate()
    nprandom_state = np.random.get_state()
            
    solutions['random_state'] = random_state
    solutions['nprandom_state'] = str(nprandom_state)
    solutions['stats'] = str(stats) # str()

    solutions['prob_gen'] = pp.prob_gen
    solutions['prob_rep'] = pp.prob_rep
    solutions['prob_decrement'] = pp.prob_decrement
    solutions['lower_bounds'] = pp.lower_bounds
    solutions['upper_bounds'] = pp.lower_bounds    
    



    stats_path = get_stats_path(path_folder, pp, num_gen)
    with open(stats_path, 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
  
