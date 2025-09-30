import numpy as np
from numba import jit
import numpy as np
import random
from copy import deepcopy
from loguru import logger

"""
    genome are a list of [2 + (nin + nout + nreg)*3]
    [beta, delta, identifiers..., enhancers..., inhibiters...]
"""

def decode_genome(genome, nin, nout):
    """
    Decode a flat genome into components.
    genome: 1D array-like: [beta, delta, identifiers..., enhancers..., inhibiters...]
    nin, nout: number of input and output proteins
    Returns: beta, delta, ids, enh, inh, n_reg (number of regulators)
    """
    genome = np.asarray(genome, dtype=np.float64)
    total = genome.size
    # print(f"total genome size : {total}")
    if (total - 2) % 3 != 0:
        # print(f"total genome size : {total}")
        raise ValueError("Genome length is not 2 + 3*n for integer n.")
    n = (total - 2) // 3  # n = nin + nout + nreg_total
    if n < (nin + nout):
        # print(f"proteins number : {n}, nin : {nin}, nout : {nout}")
        raise ValueError("Genome encodes fewer proteins than nin + nout.")
    beta = float(genome[0])
    delta = float(genome[1])
    end_ids = 2 + n
    end_enh = 2 + n * 2

    ids = genome[2 : end_ids].copy()
    enh = genome[end_ids : end_enh].copy()
    inh = genome[end_enh :].copy()
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
def genome_distance(genomeA, genomeB, nin, nout):
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

    betaA, deltaA, idsA, enhsA, inhsA, n_regA = decode_genome(genomeA, nin, nout)
    betaB, deltaB, idsB, enhsB, inhsB, n_regB = decode_genome(genomeB, nin, nout)

    la = len(genomeA)
    lb = len(genomeB)

    Dbeta = abs(betaA - betaB)/2 # normaly we should divide by betamin - betamax...
    Ddelta = abs(deltaA - deltaB)/2

    max_size = max(la, lb)

    dout = 0
    din = 0
    # compute Din and Dout 
    for i in range(nin+nout):
        idA = idsA[i]
        idB = idsB[i]
        enhA = enhsA[i]
        enhB = enhsB[i]
        inhA = inhsA[i]
        inhB = inhsB[i]
        
        dist = protein_distance(idA, enhA, inhA, idB, enhB, inhB)
        if i < nin:
            din += dist
        else:
            dout += dist

    Dreg = 0
    for i in range(nin+nout, la):
        dists = []
        for j in range(nin+nout, lb):
            idA = idsA[i]
            idB = idsB[j]
            enhA = enhsA[i]
            enhB = enhsB[j]
            inhA = inhsA[i]
            inhB = inhsB[j]
            d = protein_distance(idA, enhA, inhA, idB, enhB, inhB)
            dists.append(d)
        Dreg += min(dists)

    dist = (din + dout + Dreg + Dbeta + Ddelta)/(max_size + 2)
    return dist


def protein_distance(ids1, enh1, inh1, ids2, enh2, inh2,
                     id_coef=0.75, inh_coef=0.25, enh_coef=0.25):
    """Compute distance between protein k from genome1 and protein j from genome2
    """
    return (abs(ids1 - ids2) * id_coef +
            abs(inh1 - inh2) * inh_coef +
            abs(enh1 - enh2) * enh_coef)