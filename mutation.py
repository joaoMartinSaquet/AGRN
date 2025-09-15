import numpy as np
from genome import *
from deap import creator
from loguru import logger



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
    return individual
    

def modify_gaussian (individual, betamin, betamax, deltamin, deltamax):
    
    len_genome = len(individual)
    index = np.random.randint(0, len_genome)

    # index is beta or delta ?
    if index == 0:
        new_beta = np.random.normal() + individual[index] 
        individual[index] = np.clip(new_beta, betamin, betamax)
        # if new_beta > betamax:
        #     individual[betamax] = (new_beta - betamax) + betamin
        # elif new_beta < betamin:
        #     individual[index] = betamax - betamin - new_beta
        # else:
        #     individual[index] = new_beta
    elif index == 1:
        new_delta = np.random.normal() + individual[index]
        individual[index] = np.clip(new_delta, deltamin, deltamax)
        # if new_delta > betamax:
        #     individual[deltamax] = (new_delta - deltamax) + deltamin
        # elif new_delta < deltamin:
        #     individual[index] = deltamax - deltamin - new_delta
        # else:
        #     individual[index] = new_delta

    elif index > 1:
        new_val = np.random.normal() + individual[index]
        individual[index] = np.clip(new_val, 0., 1.)
    return individual

def add(individual, nin, nout, max_reg = 50):
    
    beta, delta, ids, enh, inh, n_reg = decode_genome(individual, nin, nout)

    if n_reg < max_reg:      
        # we add to the regulatories a random value between  0. and 1.
        ids = np.append(ids, np.random.uniform())
        enh = np.append(enh, np.random.uniform())
        inh = np.append(inh, np.random.uniform())

    genome = encode_genome(beta, delta, np.array(ids), np.array(enh), np.array(inh))

    return creator.Individual(genome.tolist())



def delete(individual, nin, nout):

    beta, delta, ids, enh, inh, n_reg = decode_genome(individual, nin, nout)
    
    if len(ids) > nin+nout:
        protein_id = np.random.randint(nin+nout, nin+nout+n_reg)
        ids = np.delete(ids, protein_id)
        enh = np.delete(enh, protein_id)
        inh = np.delete(inh, protein_id)
            
    genome = encode_genome(beta, delta, np.array(ids), np.array(enh), np.array(inh))

    return creator.Individual(genome.tolist())

    # decode_genome(individual)





def mutate(individual, nin, nout, betamin, betamax, deltamin, deltamax):
    r = np.random.random()
    if r > 0.25:
        individual = modify_gaussian(individual, betamin, betamax, deltamin, deltamax) # thisz doesnt work sry not sry see if we can fix the out of range values

    elif r > 0.5:
        individual = add(individual, nin, nout)
    else:
        individual = delete(individual, nin, nout)


    return individual,



























































































































































































































































































































































































































































































































































    return individual

def remove(individual):
    pass
