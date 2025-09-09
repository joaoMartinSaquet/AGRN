import grn
import numpy as np
import matplotlib.pyplot as plt



class grneat():

    def __init__(self, **hyperparameters):
        
        
        self.betamax = 2.
        self.betamin = 0.2
        self.deltamax = 2.
        self.deltamin = 0.2
        

    def modify(self, individual):
        """modify a random selected genes"""

        len_genome = len(individual.genome)
        index = np.random.randint(0, len_genome)

        # index is beta or delta ?
        if index == 0:
            individual.beta = np.random.random() * (self.betamax - self.betamin) + self.betamin
        elif index == 1:
            individual.delta = np.random.random() * (self.deltamax - self.deltamin) + self.deltamin
        elif index > 1:
            individual.genome[index] = np.random.random()

    def mutation(self, individual):
        """ mutation of a GRN networks
            3 mutation possible
            modify with prob = 0.25 (modifiy a random gene of a proteins)
            add a proteins with prob = 0.5
            remove a proteins with prob = 0.25
        Args:
            individual (_type_): _description_
        """
        self.modify(individual) 
        
