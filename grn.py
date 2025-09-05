import numba
import numpy as np
from numba import jit


class GRN:
    def __init__(self):
        """    # the concentrations are organised as follows:
            [nin, nout, nreg] = [x, x, *, R, R, R ] (for 2 input, 1 output and 3 regulators)
        """
        self.beta = 0
        self.delta = 0
        self.idsize = 1
        

        self.betamin = 0.2
        self.betamax = 2
        self.deltamin = 0.2
        self.deltamax = 2

        self.nin = 1
        self.nout = 1
        self.nreg = 0

        self.dt = 1

        self.size = self.nin + self.nout + self.nreg
        self.concentrations = np.zeros((self.size))
        self.identifiers = np.zeros((self.size))
        self.inhibiters = np.zeros((self.size))

        self.a  = 1
        self.f = 2
        self.setup()

    def random(self, nin = 1, nout = 1, nreg = 0):

        self.nin = nin
        self.nout = nout
        self.nreg = nreg
        self.size = self.nin + self.nout + self.nreg

        self.identifiers = np.random.random((self.nin + self.nout + self.nreg))
        self.inhibiters = np.random.random((self.nin + self.nout + self.nreg))
        self.beta = np.random.random() * (self.betamax - self.betamin) + self.betamin
        self.delta = np.random.random() * (self.deltamax - self.deltamin) + self.deltamin
        self.setup()

    def reset(self):
        """reset the concentrations of the grn
        """
        self.concentrations = 1/(self.nin + self.nout + self.nreg) * np.ones((self.nin + self.nout + self.nreg))


    def setup(self):
        self.enh_afinity_matrix, self.inh_affinity_matrix = compute_proteins_afinity(self.identifiers, self.inhibiters, self.concentrations, self.idsize, self.beta, self.a, self.f)
        self.reset()


    def __str__(self):

        dict_grn = {
            "nin": self.nin,
            "nout": self.nout,
            "nreg": self.nreg,
            "be-enh_affinity_matrix) + 1)ta": self.beta,
            "delta": self.delta,
            "idsize": self.idsize,
            "identifiers": self.identifiers,
            "concentrations": self.concentrations,
            "inhibiters": self.inhibiters,
            "a": self.a
        }

        return str(dict_grn)
    

    def step(self):
        self.concentrations = step(self.enh_afinity_matrix, self.inh_affinity_matrix, self.concentrations, self.delta, self.nin, self.nout, self.dt, self.size)
        
    def set_input(self, input_concentrations):
        self.concentrations[:self.nin] = input_concentrations

    def get_output(self):
        return self.concentrations[self.nin:self.nout+self.nin]


@jit(nopython=True)
def step(enh_afinity_matrix, inh_affinity_matrix, concentrations, delta, nin, nout, dt, nprot):

    next_concentrations = np.zeros((nprot), dtype=np.float64)
    sum_concentration = 0.0
    for i in range(nprot):
        # proteins is an input 
        if i < nin:
            next_concentrations[i] = concentrations[i]
        else:
            # prot is a regulator and regulated by either input prot our regulator proteins
            dci = 0.0
            for j in range(nprot):
                # prot is a regulator and regulated by either input prot our regulator proteins
                if (j > nin + nout - 1) or (j < nin) :
                    dci += concentrations[j] * (enh_afinity_matrix[i][j] - inh_affinity_matrix[i][j])
            dci = dci * delta/nprot
            next_concentrations[i] =  max(0.0, concentrations[i] + dt * dci)
            # sum_concentration += next_concentrations[i]

        next_concentrations = next_concentrations / sum(next_concentrations)
        # next_concentrations = next_concentrations / sum_concentration
        # next_concentrations = concentrations + delta * (np.dot(enh_afinity_matrix, concentrations) - np.dot(inh_affinity_matrix, concentrations))
        # print("sum concentration", sum_concentration)
        # if sum_concentration > 0.0:
        #     next_concentrations = next_concentrations / sum_concentration


    return next_concentrations

@jit
def compute_proteins_afinity(identifiers, inhibiters, enhancers, usize, beta, a = 0, f = 0):
    """compute afinity of all proteins, """
    matrix_size = len(identifiers)

    enhancing_match = np.zeros((matrix_size, matrix_size))
    inhibiting_match = np.zeros((matrix_size, matrix_size))

    if a != 0:
        for i in range(matrix_size):
            for j in range(matrix_size):
                enhancing_match[i][j] = usize - abs(enhancers[j] - identifiers[i])
                inhibiting_match[i][j] = usize - abs(inhibiters[j] - identifiers[i])

        
        enh_max = np.max(enhancing_match)
        inh_max = np.max(inhibiting_match)

    enh_affinity_matrix = np.zeros((matrix_size, matrix_size))
    inh_affinity_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            if a == 0:
                # afinity 0 without divided by usize but usize = 1
                enh_affinity_matrix[i][j] = -beta*abs(enhancers[j] - identifiers[i])  
                inh_affinity_matrix[i][j] = -beta*abs(inhibiters[j] - identifiers[i])
            elif a == 1:
                # affinity 1 with max outside
                enh_affinity_matrix[i][j] = beta*(enhancing_match[i][j]) - enh_max
                inh_affinity_matrix[i][j] = beta*(inhibiting_match[i][j]) - inh_max
            else: # a == 2
                # affinity 2 with max  inside
                enh_affinity_matrix[i][j] = beta*(enhancing_match[i][j] - enh_max ) 
                inh_affinity_matrix[i][j] = beta*(inhibiting_match[i][j] - inh_max ) 

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