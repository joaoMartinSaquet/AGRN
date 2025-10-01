


class LoggerDumper():
    def __init__(self):
        self.logdict = {"generation": [], "best_indiv": [], "best_fitness": [], "avg_fitness": [], "min_fitness": [], "max_fitness": []}

    def update(self, generations, evaluations):