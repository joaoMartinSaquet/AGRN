from deap import creator, base, tools, algorithms
import numpy as np
from loguru import logger
import json

class Statistics:
    def __init__(self):
        self.statDict = {"generation": [], "best_indiv": [], 
                        "best_fitness": [], "avg_fitness": [], 
                        "min_fitness": [], "best_length": []}

    def update(self, generation, best_indiv, best_fitness, avg_fitness, min_fitness, best_length):
        self.statDict["generation"].append(generation)
        self.statDict["best_indiv"].append(best_indiv)
        self.statDict["best_fitness"].append(best_fitness)
        self.statDict["avg_fitness"].append(avg_fitness)
        self.statDict["min_fitness"].append(min_fitness)
        self.statDict["best_length"].append(best_length)

    def print_header(self):
        stat_string = f""
        for fields in self.statDict:
            stat_string += f"{fields} | \t"
        
        logger.info(stat_string)
        logger.info("-" * 70)


    def print(self):
        stat_string = f""
        for fields in self.statDict:
            if fields == "best_indiv":
                stat_string += f" | \t"
            else:
                stat_string += f"{self.statDict[fields][-1]} | \t"
            # logger.info(f"{fields} : {self.statDict[fields]}")
        logger.info(stat_string)

    def clear(self):
        for fields in self.statDict:    self.statDict[fields].clear()    

    def dump(self, file_path, file_name):
        if ".json" not in file_name:    file_name += ".json"
        with open(file_path + file_name, "w") as f:
            json.dump(self.statDict, f)

    def compute_stats(self, pop, gen):
        """
        Compute statistics for a population and return a formatted string.
        - Fitness: avg, min, max, std, median
        - Best genome length: length of individual with best fitness
        """

        # Extract fitness values
        fitness_vals = [ind.fitness.values[0] for ind in pop]

        # Identify the best individual (lowest fitness for minimization)
        best_idx = np.argmax(fitness_vals)
        best_ind = pop[best_idx]

        avg_fit = np.mean(fitness_vals)
        min_fit = np.min(fitness_vals)
        max_fit = np.max(fitness_vals)
        std_fit = np.std(fitness_vals)
        median_fit = np.median(fitness_vals)

        # Genome length of best individual
        best_len = len(best_ind)

        self.update(gen, best_ind, max_fit, avg_fit, min_fit, best_len)

        return {
            "avg": np.mean(fitness_vals),
            "min": np.min(fitness_vals),
            "max": np.max(fitness_vals),
            "std": np.std(fitness_vals),
            "median": np.median(fitness_vals),
            "best_len": len(best_ind),
        }

        
def best_genome_length(population):
    print("population : ", population)
    hof = tools.selBest(population, k=1)
    return len(hof[0])




def format_stats(gen, stats: dict) -> str:
    """Format stats dict into a table-like string."""
    return (f"{gen:>3} | "
            f"{stats['avg']:>10.3f} | "
            f"{stats['min']:>10.3f} | "
            f"{stats['max']:>10.3f} | "
            f"{stats['std']:>8.3f} | "
            f"{stats['median']:>10.3f} | "
            f"{stats['best_len']:>5}")



