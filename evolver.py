import grn
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from deap import creator, base, tools, algorithms
from functools import partial 
from loguru import logger
import seaborn as sns
# sns.set_style("darkgrid")  # Seaborn style

from mutation import *
from genome import *
from crossover import *

import multiprocessing

def evaluate(individual, problem):
    fit, _ = problem.eval(individual)
    return fit,
    



class EATMuPlusLambda():

    def __init__(self, nin, nout, nreg,
                 beta_min=0.2, beta_max=2, delta_min=0.2, delta_max=2,
                 crossover_threshold = 0.5, mutation_rate = 0.25, crossover_rate = 0.5,
                 mut_add_prob = 0.5, mut_del_prob = 0.25, mut_mod_prob = 0.25):
        
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

        self.init_deap(nin=nin, nout=nout, nreg=nreg)

    def init_deap(self, nin, nout, nreg, scoop=False):

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
        # self.toolbox.register("mate", cx, nin=nin, nout=nout)  # Crossover (blend parents) aligned crossover
        self.toolbox.register("mate", tools.cxBlend, alpha=10)  # Crossover (blend parents) { doesnt  work at his get some negative and outside the range values}
        # self
        
        self.toolbox.register("mutate", mutate, nin=nin, nout=nout, betamin=self.betamin, betamax=self.betamax, deltamin=self.deltamin, deltamax=self.deltamax)  # Mutation
        # toolbox.register("mutate", lambda individual: tools.mutGaussian(individual, mu=0, sigma=0.5, indpb=0.5))  # Mutation

        # self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
        self.toolbox.register("select", tools.selSPEA2)  # Selection
        
    

        # population = self.toolbox.population(n=500)  # 50 individuals
        # hof = tools.HallOfFame(1)  # Track best individual

        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        self.stats_fit.register("avg", np.mean)
        self.stats_fit.register("min", np.min)
        self.stats_fit.register("max", np.max)
        self.stats_fit.register("std", np.std)
        self.stats_fit.register("median", np.median)
        



    def run(self,n_gen, problem, mu, lambda_, cxpb = 0.0, mutpb = 1.0, multiproc = False, verbose=True):
        

        self.toolbox.register("evaluate", problem.eval)  # Fitness = sum of genome values
        
        if multiproc:
            pool = multiprocessing.Pool()
            self.toolbox.register("map", pool.map)
            
        population = self.toolbox.population(n=500)  # 50 individuals
        hof = tools.HallOfFame(1)  # Track best individual

        self.pop, self.logbook = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            lambda_ = lambda_,
            mu=mu,
            cxpb=cxpb,       # Crossover probability
            mutpb=mutpb,      # Mutation probability
            ngen=n_gen,        # Generations
            halloffame=hof,
            stats=self.stats_fit,
            verbose=verbose    # Print progress
        )

        # self.pop, self.logbook = algorithms.eaMuCommaLambda(
        #     population,
        #     self.toolbox,
        #     lambda_ = lambda_,
        #     mu=mu,
        #     cxpb=cxpb,       # Crossover probability
        #     mutpb=mutpb,      # Mutation probability
        #     ngen=n_gen,        # Generations
        #     halloffame=hof,
        #     stats=self.stats_fit,
        #     verbose=verbose    # Print progress
        # )


        return hof,


    def visualize_evolutions(self):
        avg_fitness = [r['avg'] for r in self.logbook]
        max_fitness = [r['max'] for r in self.logbook]
        min_fitness = [r['min'] for r in self.logbook]
        std_fitness = [r['std'] for r in self.logbook]


        # Example arrays from your logbook
        # avg_fitness, max_fitness, min_fitness, std_fitness

        gens = np.arange(len(avg_fitness))

        fig, ax = plt.subplots(figsize=(10, 6))

        # Fill area for standard deviation around the mean
        ax.fill_between(gens,
                        np.array(avg_fitness) - np.array(std_fitness),
                        np.array(avg_fitness) + np.array(std_fitness),
                        color='gray', alpha=0.3, label='Std Dev')

        # Plot avg, max, min with different colors and line styles
        ax.plot(gens, avg_fitness, color='blue', linewidth=2, label='Average Fitness')
        ax.plot(gens, max_fitness, color='green', linewidth=2, label='Max Fitness')
        # ax.plot(gens, min_fitness, color='red', linewidth=2, label='Min Fitness')

        # Labels, title, legend
        ax.set_xlabel('Generation', fontsize=14)
        ax.set_ylabel('Fitness', fontsize=14)
        ax.set_title('Evolution of Fitness over Generations', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Optional: tighter layout
        plt.tight_layout()
        plt.show()