import numba
import numpy as np
from numba import jit
import loguru as logger

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
        self.enhancers = np.zeros((self.size))
        self.enh_afinity_matrix = np.zeros((self.size, self.size))
        self.inh_affinity_matrix = np.zeros((self.size, self.size))

        self.a  = 0
        self.f = 0
        self.setup()

    def __init__(self, nin = 1, nout = 1, nreg = 0, a = 0, f = 0):
        """init with random parameters"""
        self.a = a
        self.f = f
        self.beta = 0
        self.delta = 0
        self.idsize = 1
        self.dt = 1

        self.betamin = 0.2
        self.betamax = 2
        self.deltamin = 0.2
        self.deltamax = 2
        self.random(nin, nout, nreg)


    def random(self, nin = 1, nout = 1, nreg = 0):

        self.nin = nin
        self.nout = nout
        self.nreg = nreg
        self.size = self.nin + self.nout + self.nreg

        self.identifiers = np.random.random((self.nin + self.nout + self.nreg))
        self.inhibiters = np.random.random((self.nin + self.nout + self.nreg))
        self.enhancers = np.random.random((self.nin + self.nout + self.nreg))
        self.beta = np.random.random() * (self.betamax - self.betamin) + self.betamin
        self.delta = np.random.random() * (self.deltamax - self.deltamin) + self.deltamin
        self.setup()

        return self.genome
    

    def reset(self):
        """reset the concentrations of the grn
        """
        self.concentrations = 1/(self.size) * np.ones((self.size))
    
    def warmup(self, nsteps=25):
        self.set_input(np.zeros((self.nin)))
        self.step(nsteps)

    def setup(self):
        self.set_genome()
        self.enh_afinity_matrix, self.inh_affinity_matrix = compute_proteins_afinity(self.identifiers, self.enhancers, self.inhibiters, self.idsize, self.beta, self.a, self.f)
        # print(self.enh_afinity_matrix)
        # print(self.inh_affinity_matrix)
        self.reset()

    
    def set_genome(self):
        """
        set the genome of the grn as a list
        """
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

        all_values = []
        for key, value in self.dict_grn.items():
            if isinstance(value, np.ndarray):
                all_values.extend(value.tolist())
                # print("value : ", value)
                # for el in value.tolist():  # Handle NumPy arrays
                    # all_values.extend(el)  # Convert array to list and extend
            else:  # Handle scalar values
                all_values.append(value)

        self.genome = all_values
        return all_values
    
    def __str__(self):
        return str(self.dict_grn)
    
    def step(self, nsteps = 1):

        for i in range(nsteps):
            self.concentrations = step(self.enh_afinity_matrix, self.inh_affinity_matrix, self.concentrations, self.delta, self.nin, self.nout, self.dt, self.size)
        
    def set_input(self, input_concentrations):
        self.concentrations[:self.nin] = input_concentrations

    def get_output(self):
        return self.concentrations[self.nin:self.nout+self.nin].copy()

    def from_genome(self, genome):
        """genome of the grn

        the genome is an array of length 8 + 3 * N
        with N = nin + nout + nreg

        Args:
            genome (_type_): [nin, nout, nreg, beta, delta, a, f, idsize, identifiers, enhancers, inhibiters] ]
                             [intn int,  int,  float, float, int, int, float, float[], float[], float[]]
                

        Returns:
            _type_: _description_
        """
        self.nin = genome[0]
        self.nout = genome[1]
        self.nreg = genome[2]
        self.beta = genome[3]
        self.delta = genome[4]
        self.a = genome[5]
        self.f = genome[6]
        self.idsize = genome[7]

        self.size = self.nin + self.nout + self.nreg
        id_start, id_end = compute_identifiers_genome_start_end_index(self.size)
        enh_start, enh_end = compute_enhancers_genome_start_end_index(id_end, self.size)
        inh_start, inh_end = compute_inhibiters_genome_start_end_index(enh_end, self.size)

        self.identifiers = genome[id_start:id_end]
        self.enhancers = genome[enh_start:enh_end]
        self.inhibiters = genome[inh_start:inh_end]
        self.setup()






@jit(nopython=True)
def step(enh_afinity_matrix, inh_affinity_matrix, concentrations, delta, nin, nout, dt, nprot):

    next_concentrations = np.zeros((nprot), dtype=np.float64)
    # sum_concentration = 0.0

    
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
                if (j > nin + nout ) or (j < nin) :
                    
                    enhancing_factor += concentrations[j] * enh_afinity_matrix[j][i]
                    inhibiting_factor += concentrations[j] * inh_affinity_matrix[j][i]
                    # enhancing_factor += concentrations[j] * enh_afinity_matrix[j][i]
                    # inhibiting_factor += concentrations[j] * inh_affinity_matrix[j][i]
                    # dci += concentrations[j] * (enh_afinity_matrix[i][j] - inh_affinity_matrix[i][j])
            dci = delta * (enhancing_factor - inhibiting_factor)/(nprot)     
            
            next_concentrations[i] = min(1.0,max(0.0, concentrations[i] + dt * dci))            # sum_concentration += next_concentrations[i]

    # sum_concentration = np.sum(next_concentrations[nin:])
    # if sum_concentration > 0.0:
    #     next_concentrations[nin:] = next_concentrations[nin:] / sum_concentration
        
        
        # next_concentrations = next_concentrations / sum_concentration
        # next_concentrations = concentrations + delta * (np.dot(enh_afinity_matrix, concentrations) - np.dot(inh_affinity_matrix, concentrations))
        # print("sum concentration", sum_concentration)
        # if sum_concentration > 0.0:
        #     next_concentrations = next_concentrations / sum_concentration


    return next_concentrations.copy()

@jit
def compute_proteins_afinity(identifiers, enhancers, inhibiters, usize, beta, a = 0, f = 0):
    """compute afinity of all proteins, """
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
                # afinity 0 without divided by usize but usize = 1
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

