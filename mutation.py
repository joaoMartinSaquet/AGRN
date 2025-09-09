import numpy as np


def modify(individual, betamin, betamax, deltamin, deltamax):
    """modify a random selected genes"""
    
    len_genome = len(individual)
    index = np.random.randint(0, len_genome)
    # index is beta or delta ?
    if index == 0:
        individual[index] = np.random.uniform() * (betamax - betamin) + betamin
    elif index == 1:
        individual[index] = np.random.uniform() * (deltamax - deltamin) + deltamin
    elif index > 1:
        individual[index] = np.random.uniform()
    return individual, 
    
def add(individual):
    pass

def remove(individual):
    pass
