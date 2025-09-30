from deap import creator, base, tools, algorithms
import numpy as np


class MultiStatistics:
    def __init__(self, *stats):
        self.stats = stats
        self.fields = []
        for stat in stats:
            self.fields.extend(stat.fields)

    def compile(self, population):
        result = []
        for stat in self.stats:
            result.extend(stat.compile(population))
        return result


def best_genome_length(population):
    print("population : ", population)
    hof = tools.selBest(population, k=1)
    return len(hof[0])




def compute_stats(pop):
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


    return {
        "avg": np.mean(fitness_vals),
        "min": np.min(fitness_vals),
        "max": np.max(fitness_vals),
        "std": np.std(fitness_vals),
        "median": np.median(fitness_vals),
        "best_len": len(best_ind),
    }

def format_stats(gen, stats: dict) -> str:
    """Format stats dict into a table-like string."""
    return (f"{gen:>3} | "
            f"{stats['avg']:>10.3f} | "
            f"{stats['min']:>10.3f} | "
            f"{stats['max']:>10.3f} | "
            f"{stats['std']:>8.3f} | "
            f"{stats['median']:>10.3f} | "
            f"{stats['best_len']:>5}")
