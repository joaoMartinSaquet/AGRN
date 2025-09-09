import grn
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from mutation import *
from genome import *

class grneat():

    def __init__(self, **hyperparameters):
        
        
        self.betamax = 2.
        self.betamin = 0.2
        self.deltamax = 2.
        self.deltamin = 0.2

        self.nin = 1
        self.nout = 1

        self.crossover_threshold = 0.5
        


    def mutation(self, individual):
        """ mutation of a GRN networks
            3 mutation possible
            modify with prob = 0.25 (modifiy a random gene of a proteins)
            add a proteins with prob = 0.5
            remove a proteins with prob = 0.25
        Args:
            individual (_type_): _description_
        """
        
        new_ind = individual
        # need to draw a random individual
        individual = self.modify(new_ind) 

        return individual,
        

    def crossover_operator(self, individual1, individual2):
        """Cross the individual GRNs to create a child GRN
        The child inherits input and output proteins randomly from both individuals
        Regulatory genes are passed on based on alignment;
        the closest genes from each individual are selected between for inheritance.
        Beta and delta are selected randomly from both individuals.
        """
       
        return crossover(individual1, individual2, self.nin, self.nout)