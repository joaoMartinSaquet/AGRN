from . import grn
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from deap import creator, base, tools, algorithms
from functools import partial 
from loguru import logger
import seaborn as sns
from joblib import Parallel, delayed



# sns.set_style("darkgrid")  # Seaborn style

from .mutation import *
from .genome import *
from .crossover import *
from .statistics import *

import multiprocessing

# def evaluate(individual, problem):
#     fit, _ = problem.eval(individual)
#     return fit,
    
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
        self.toolbox.register("mate", cx, nin=nin, nout=nout)  # Crossover (blend parents) aligned crossover
        # self.toolbox.register("mate", tools.cxBlend, alpha=10)  # Crossover (blend parents) { doesnt  work at his get some negative and outside the range values}
        # self
        
        self.toolbox.register("mutate", mutate, nin=nin, nout=nout, betamin=self.betamin, betamax=self.betamax, deltamin=self.deltamin, deltamax=self.deltamax)  # Mutation
        # toolbox.register("mutate", lambda individual: tools.mutGaussian(individual, mu=0, sigma=0.5, indpb=0.5))  # Mutation

        # self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
        self.toolbox.register("select", tools.selSPEA2)  # Selection
        
        self.stats = Statistics()
        # self.mstats = MultiStatistics(stats_fit, stats_best_len)

    def run(self,n_gen, eval_fun, mu, lambda_, cxpb = 0.25, mutpb = 0.75, comma = False, log_path="..", log_name = "stat_history", backend = 'threading', multiproc = False, n_proc = None, verbose=True):
        

        self.toolbox.register("evaluate", eval_fun)  # Fitness = sum of genome values
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "avg", "min", "max", "std", "median", "best_len"]

            
        population = self.toolbox.population(n=mu)  # 50 individuals
        hist = tools.History()
        hof = tools.HallOfFame(1)  # Track best individual
        hist.update(population)

        pop = self.toolbox.population(n=mu)
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        
        fitnesses = Parallel(n_jobs=n_proc, backend=backend)(
                delayed(self.toolbox.evaluate)(genome) for genome in invalid_ind)
            
            
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        self.stats.compute_stats(pop, -1)
        self.stats.print_header()
        self.stats.print()

        hist.update(pop)
        for gen in range(n_gen):
            # Variation
            offspring = algorithms.varOr(pop, self.toolbox, lambda_, cxpb, mutpb)

            # Evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = Parallel(n_jobs=n_proc, backend=backend)(
                delayed(self.toolbox.evaluate)(genome) for genome in invalid_ind)
            
            
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if not comma:
                # we select from the mu and pop we then maintain high competitiveness
                pop[:] = self.toolbox.select(pop + offspring, mu)

            # we select from the lambda to maintain the diversity maybe useless if we have the speciations
            else:
                pop[:] = self.toolbox.select(offspring, mu)
            hist.update(pop)
            
            best_ind = tools.selBest(pop, 1)[0]
            hof.update(pop)

            
            self.stats.compute_stats(pop, gen)
            self.stats.print()

        self.stats.dump(log_path, "stats.json")
        return hof, hist 



    def visualize_evolutions(self):
        avg_fitness = np.array(self.stats.statDict['avg_fitness'])
        max_fitness = np.array(self.stats.statDict['best_fitness'])
        min_fitness = np.array(self.stats.statDict['min_fitness'])


        eps = np.max(max_fitness) - np.min(min_fitness)
        # Example arrays from your logbook
        # avg_fitness, max_fitness, min_fitness, std_fitness

        gens = np.arange(len(avg_fitness))

        fig, ax = plt.subplots(figsize=(10, 6))

        # # Fill area for standard deviation around the mean
        # ax.fill_between(gens,
        #                max_fitness,
        #                min_fitness,
        #                 color='gray', alpha=0.3, label='Std Dev')

        # Plot avg, max, min with different colors and line styles
        ax.plot(gens, avg_fitness, color='blue', linewidth=2, label='Average Fitness')
        ax.plot(gens, max_fitness, color='green', linewidth=2, label='Max Fitness')
        # ax.plot(gens, min_fitness, color='red', linewidth=2, label='Min Fitness')

        # Labels, title, legend
        ax.set_xlabel('Generation', fontsize=14)
        ax.set_ylabel('Fitness', fontsize=14)
        ax.set_title('Evolution of Fitness over Generations', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        
        ax.grid(True, which='both')   
        # Optional: tighter layout
        plt.tight_layout()
        plt.show()

    
    def dumps_logs(self):


        return self.logbook