import grn
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from deap import creator, base, tools, algorithms
from functools import partial 
from loguru import logger


from mutation import *
from genome import *
from crossover import *

class EATMuPlusLambda():

    def __init__(self,beta_min=0.2, beta_max=2, delta_min=0.2, delta_max=2,
                 crossover_threshold = 0.5, mutation_rate = 0.25, crossover_rate = 0.5,
                 mut_add_prob = 0.5, mut_del_prob = 0.25, mut_mod_prob = 0.25,):
        
        # grn parameters

        self.betamax = beta_min
        self.betamin = beta_max
        self.deltamax = delta_max
        self.deltamin = delta_min



        self.crossover_threshold = crossover_threshold

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.mut_add_prob = mut_add_prob
        self.mut_del_prob = mut_del_prob
        self.mut_mod_prob = mut_mod_prob

    def init_deap(self, nin, nout, nreg):

        self.nin = nin
        self.nout = nout
        self.nreg = nreg


        # problem = p.FrenchFlagProblem(nin=nin, nout=nout, nreg=nreg)


        # 1. Define the fitness and individual types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)


        def init_indiv(nin, nout, start_nreg):
            return random_genome(nin, nout, start_nreg)

        init_indiv_fun = partial(init_indiv, nin=nin, nout=nout, start_nreg=nreg)
        # 2. Set up the toolbox (genetic operators)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual",tools.initIterate, creator.Individual,init_indiv_fun,)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 3. Define genetic operators
        self.toolbox.register("mate", cx, nin=nin, nout=nout)  # Crossover (blend parents)
        self.toolbox.register("mutate", mutate, nin=nin, nout=nout, betamin=self.betamin, betamax=self.betamax, deltamin=self.deltamin, deltamax=self.deltamax)  # Mutation
        # toolbox.register("mutate", lambda individual: tools.mutGaussian(individual, mu=0, sigma=0.5, indpb=0.5))  # Mutation

        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
    

        # population = self.toolbox.population(n=500)  # 50 individuals
        # hof = tools.HallOfFame(1)  # Track best individual

        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        self.stats_fit.register("avg", np.mean)
        self.stats_fit.register("min", np.min)
        self.stats_fit.register("max", np.max)
        self.stats_fit.register("std", np.std)
        

    def run(self, problem, n_gen=100):
        
        def evaluate(individual):
            fit, _ = problem.eval(individual)
            return fit,


        self.toolbox.register("evaluate", evaluate)  # Fitness = sum of genome values

        population = self.toolbox.population(n=500)  # 50 individuals
        hof = tools.HallOfFame(1)  # Track best individual

        self.alg = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            lambda_ = 500,
            mu=100,
            cxpb=0.2,       # Crossover probability
            mutpb=0.8,      # Mutation probability
            ngen=200,        # Generations
            halloffame=hof,
            stats=stats_fit,
            verbose=True    # Print progress
        )