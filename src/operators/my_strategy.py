import copy
from math import sqrt, log, exp
import numpy
import math
import deap.tools as tools




class EDA_Strategy(object): #EDA(object):
    def __init__(self, centroid, sigma, mu, lambda_):
        self.dim = len(centroid)
        self.loc = numpy.array(centroid)
        self.sigma = numpy.array(sigma)
        self.lambda_ = lambda_
        self.mu = mu
    
    def generate(self, ind_init, lower, upper):
        # Generate lambda_ individuals and put them into the provided class
        arz = self.sigma * numpy.random.randn(self.lambda_, self.dim) + self.loc
        arz = numpy.round(arz) #arz = numpy.round(arz) .astype(int).tolist()
        arz = numpy.clip(arz, lower, upper)
        #return numpy.round(list(map(ind_init, arz)))
        return list(map(ind_init, arz))
    
    def update(self, population, lower, upper):
        # Sort individuals so the best is first
        #sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)
        population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # Compute the average of the mu best individuals
        z = population[:self.mu] - self.loc
        avg = numpy.mean(z, axis=0)
        
        # Adjust variances of the distribution
        self.sigma = numpy.sqrt(numpy.sum((z - avg)**2, axis=0) / (self.mu - 1.0))
        self.loc = self.loc + avg
