import numpy as np
from numba import jit
import numpy as np
import random
from copy import deepcopy

"""
    genome are a list of [2 + (nin + nout + nreg)*3]
    [beta, delta, identifiers..., enhancers..., inhibiters...]
"""
@jit
def decode_genome(genome, nin, nout):
    """
    Decode a flat genome into components.
    genome: 1D array-like: [beta, delta, identifiers..., enhancers..., inhibiters...]
    nin, nout: number of input and output proteins
    Returns: beta, delta, ids, enh, inh, n_reg (number of regulators)
    """

    genome = np.asarray(genome, dtype=np.float64)
    total = genome.size
    if (total - 2) % 3 != 0:
        print(total)
        raise ValueError("Genome length is not 2 + 3*n for integer n.")
    n = (total - 2) // 3  # n = nin + nout + nreg_total
    if n < (nin + nout):
        raise ValueError("Genome encodes fewer proteins than nin + nout.")
    beta = float(genome[0])
    delta = float(genome[1])
    ids = genome[2 : 2 + n].copy()
    enh = genome[2 + n : 2 + 2 * n].copy()
    inh = genome[2 + 2 * n : 2 + 3 * n].copy()
    n_reg = n - (nin + nout)
    return beta, delta, ids, enh, inh, n_reg


def encode_genome(beta, delta, ids, enh, inh):
    """Reassemble genome"""
    return np.concatenate(([beta, delta], ids, enh, inh), dtype=np.float64)


@jit
def random_genome(nin, nout, nreg, beta_min=0.2, beta_max=2, delta_min=0.2, delta_max=2):
    genome = np.random.random(2 + (nin + nout + nreg) * 3)
    genome[0] = genome[0] * (beta_max - beta_min) + beta_min
    genome[1] = genome[1] * (delta_max - delta_min) + delta_min
    
    return genome


@jit
def protein_distance(genomeA, genomeB, k, j,
                     id_coef=1.0, inh_coef=1.0, enh_coef=1.0):
    """
    Compute distance between protein k of genomeA and protein j of genomeB.
    
    Genome format:
    [beta, delta, identifiers..., enhancers..., inhibitors...]
    
    Args:
        genomeA, genomeB : list or array
        k, j             : protein indices (0 .. nin+nout+nreg-1)
        id_coef, inh_coef, enh_coef : distance coefficients
    
    Returns:
        float : distance
    """
    # extract sizes
    n = (len(genomeA) - 2) // 3   # total number of proteins in genomeA
    m = (len(genomeB) - 2) // 3   # total number of proteins in genomeB
    
    # --- decode genomeA ---
    idsA   = genomeA[2:2+n]
    enhA   = genomeA[2+n:2+2*n]
    inhA   = genomeA[2+2*n:2+3*n]
    
    # --- decode genomeB ---
    idsB   = genomeB[2:2+m]
    enhB   = genomeB[2+m:2+2*m]
    inhB   = genomeB[2+2*m:2+3*m]
    
    # --- compute distance ---
    dist = (abs(idsA[k] - idsB[j]) * id_coef +
            abs(inhA[k] - inhB[j]) * inh_coef +
            abs(enhA[k] - enhB[j]) * enh_coef)
    return dist