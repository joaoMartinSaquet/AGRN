import numba
import numpy as np
from numba import jit
import loguru as logger
from .genome import *

class GRN:

    dt = 1
    nin = 1
    nout = 1
    nreg = 0
    idsize = 1.
    betamin = 0.2
    betamax = 2
    deltamin = 0.2
    deltamax = 2
    a = 0
    f = 1

    identifiers = []
    enhancers = []
    inhibiters = []
    concentrations = []

    def __init__(self,):
        """    # the concentrations are organised as follows:
            [nin, nout, nreg] = [x, x, *, R, R, R ] (for 2 input (x), 1 output(*) and 3 regulators)
        """
        self.random(self.nin, self.nout, self.nreg)

        self.setup()

    def __init__(self, genome : list, nin = 1, nout = 1):
        """init with genomes"""
        self.genome = genome
        self.nin = nin
        self.nout = nout
        self.beta, self.delta, self.identifiers, self.enhancers, self.inhibiters, self.nreg = decode_genome(genome, nin, nout)
        self.size = self.nin + self.nout + self.nreg
        self.setup()
        

    def random(self, nin = 1, nout = 1, nreg = 0):
        """init with random parameters"""

        self.nin = nin
        self.nout = nout
        self.nreg = nreg
        self.size = self.nin + self.nout + self.nreg

        self.identifiers = np.random.random((self.nin + self.nout + self.nreg))
        self.inhibiters = np.random.random((self.nin + self.nout + self.nreg))
        self.enhancers = np.random.random((self.nin + self.nout + self.nreg))
        self.beta = np.random.random() * (self.betamax - self.betamin) + self.betamin
        self.setup()

        return encode_genome(self.beta, self.delta, self.identifiers, self.enhancers, self.inhibiters)
    

    def reset(self):
        """reset the concentrations of the grn
        """
        self.concentrations = 1/(self.size) * np.ones((self.size))
    
    def warmup(self, nsteps=25):
        self.set_input(np.zeros((self.nin)))
        self.step(nsteps)

    def setup(self):
        self.enh_affinity_matrix, self.inh_affinity_matrix = compute_proteins_affinity(self.identifiers, self.enhancers, self.inhibiters, self.idsize, self.beta, self.a, self.f)
        # print(self.enh_affinity_matrix)
        # print(self.inh_affinity_matrix)
        self.reset()


    
    def __str__(self):

        self.dict_grn = {
            "nin": self.nin,
            "nout": self.nout,
            "nreg": self.nreg,
            "beta": self.beta,
            "delta": self.delta,
            "a": self.a,
            "f": self.f,
            "idsize": self.idsize,
            "identifiers": self.identifiers,
            "enhancers" : self.enhancers,
            "inhibiters": self.inhibiters,
        }

        return str(self.dict_grn)
    
    def step(self, nsteps = 1):

        for i in range(nsteps):
            self.concentrations = step(self.enh_affinity_matrix, self.inh_affinity_matrix, self.concentrations, self.delta, self.nin, self.nout, self.dt, self.size)
        
    def set_input(self, input_concentrations):
        self.concentrations[:self.nin] = input_concentrations

    def get_output(self):
        """ Do a pairwise concentration difference of outputs proteins : out0 = (Co0 - Co1)/(Co0 + Co1)
        """
        output_concentrations = self.concentrations[self.nin:self.nout+self.nin].copy()
        # lo = len(output_concentrations)

        # if lo%2 != 0: logger.error("Number of outputs is not even")

        # out = np.zeros((lo//2))
        # j = 0
        # for i in range(0, lo, 2):
        #     if  (output_concentrations[i] + output_concentrations[i+1]) == 0.0:
        #         o = 1.0   
        #     else:
        #         o = (output_concentrations[i] - output_concentrations[i+1]) / (output_concentrations[i] + output_concentrations[i+1])             
        #     out[j] = np.clip(o, 0.0, 1.0)
        #     j += 1

        # # out = output_concentrations
        return output_concentrations
    


@jit(nopython=True, cache=True)
def step(enh_affinity_matrix, inh_affinity_matrix, concentrations, delta, nin, nout, dt, nprot):

    next_concentrations = np.zeros((nprot), dtype=np.float64)
    # sum_concentration = 0.0

    # proteins i is the protein that is getting regulated
    for i in range(nprot): 
        # proteins is an input  
        if i < nin:
            next_concentrations[i] = concentrations[i]
        else:
            # prot is a regulator and regulated by either input prot our regulator proteins
            # dci = 0.0


            enhancing_factor = 0.0
            inhibiting_factor = 0.0
            for j in range(nprot):
                # prot is a regulator and regulated by either input prot our regulator proteins
                if (j > nin + nout-1) or (j < nin) :
                    
                    enhancing_factor += concentrations[j] * enh_affinity_matrix[j][i]
                    inhibiting_factor += concentrations[j] * inh_affinity_matrix[j][i]
                    # enhancing_factor += concentrations[j] * enh_affinity_matrix[j][i]
                    # inhibiting_factor += concentrations[j] * inh_affinity_matrix[j][i]
                    # dci += concentrations[j] * (enh_affinity_matrix[i][j] - inh_affinity_matrix[i][j])
            dci = delta * (enhancing_factor - inhibiting_factor)/(nprot)     
            
            next_concentrations[i] = min(1.0,max(0.0, concentrations[i] + dt * dci))            # sum_concentration += next_concentrations[i]
        
        # # regularization steps 
        # sum_concentrations = np.sum(next_concentrations[nin:])
        # if sum_concentrations > 0.0:
        #     next_concentrations[nin:] = next_concentrations[nin:] / sum_concentrations

    return next_concentrations

@jit(cache=True, nopython=True)
def compute_proteins_affinity(identifiers, enhancers, inhibiters, usize, beta, a = 0, f = 0):
    """compute affinity of all proteins, """
    matrix_size = len(identifiers)

    enhancing_match = np.zeros((matrix_size, matrix_size))
    inhibiting_match = np.zeros((matrix_size, matrix_size))


    for i in range(matrix_size):
        for j in range(matrix_size):
            enhancing_match[i][j] = usize - abs(enhancers[i] - identifiers[j])
            inhibiting_match[i][j] = usize - abs(inhibiters[i] - identifiers[j])
    
    if a != 0:
        enh_max = np.max(enhancing_match)
        inh_max = np.max(inhibiting_match)
    

    enh_affinity_matrix = np.zeros((matrix_size, matrix_size))
    inh_affinity_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            if a == 0:
                # affinity 0 without divided by usize but usize = 1
                enh_affinity_matrix[i][j] = -beta*enhancing_match[i][j]  
                inh_affinity_matrix[i][j] = -beta*inhibiting_match[i][j]
            elif a == 1:
                # affinity 1 with max outside
                enh_affinity_matrix[i][j] = beta*(1.0 - enhancing_match[i][j]) - enh_max
                inh_affinity_matrix[i][j] = beta*(1.0 - inhibiting_match[i][j]) - inh_max
            else: # a == 2
                # affinity 2 with max  inside
                enh_affinity_matrix[i][j] = beta*(1.0 - enhancing_match[i][j] - enh_max ) 
                inh_affinity_matrix[i][j] = beta*(1.0 - inhibiting_match[i][j] - inh_max ) 

    if f == 0:
        enh_affinity_matrix = np.exp(enh_affinity_matrix)
        inh_affinity_matrix = np.exp(inh_affinity_matrix)
    if f == 1:
        enh_affinity_matrix = np.tanh(enh_affinity_matrix) + 1
        inh_affinity_matrix = np.tanh(inh_affinity_matrix) + 1
    if f == 2:
        enh_affinity_matrix = 2/(np.exp(-enh_affinity_matrix) + 1)
        inh_affinity_matrix = 2/(np.exp(-inh_affinity_matrix) + 1)


    return enh_affinity_matrix, inh_affinity_matrix


def compute_identifiers_genome_start_end_index(N):
    start = 8
    end = start + N
    return start, end

def compute_enhancers_genome_start_end_index(identifiers_end, N):
    start = identifiers_end
    end = start + N
    return start, end

def compute_inhibiters_genome_start_end_index(enhancers_end, N):
    start = enhancers_end
    end = start + N
    return start, end

