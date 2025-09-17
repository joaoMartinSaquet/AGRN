from deap import creator, base, tools, algorithms

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