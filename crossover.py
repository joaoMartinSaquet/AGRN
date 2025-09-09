from deap import creator
import random, numpy as np
from copy import deepcopy
from genome import decode_genome, encode_genome

def cx(ind1, ind2, nin, nout, threshold=0.1):
    """
    Custom crossover for genomes encoded as [beta, delta, ids, enh, inh].
    Returns two DEAP Individuals.
    """
    # --- Decode parents ---
    beta1, delta1, ids1, enh1, inh1, nreg1 = decode_genome(ind1, nin, nout)
    beta2, delta2, ids2, enh2, inh2, nreg2 = decode_genome(ind2, nin, nout)

    n1, n2 = len(ids1), len(ids2)

    # Function to create one child
    def make_child(idsA, enhA, inhA, betaA, deltaA,
                   idsB, enhB, inhB, betaB, deltaB):
        child_ids, child_enh, child_inh = [], [], []

        # 1. Copy I/O proteins
        for k in range(nin + nout):
            if np.random.randint(2) == 0:
                child_ids.append(idsA[k]); child_enh.append(enhA[k]); child_inh.append(inhA[k])
            else:
                child_ids.append(idsB[k]); child_enh.append(enhB[k]); child_inh.append(inhB[k])

        # 2. Regulatory proteins
        pA_range = list(range(nin+nout, len(idsA)))
        pB_range = list(range(nin+nout, len(idsB)))
        random.shuffle(pA_range)
        random.shuffle(pB_range)
        pA_remaining = deepcopy(pA_range)

        pA_count, pB_count = 0, 0
        for i in pA_range:
            min_dist, paired = threshold, None
            for j in pB_range:
                d = protein_distance_single(idsA, enhA, inhA,
                                            idsB, enhB, inhB, i, j)
                if d < min_dist:
                    min_dist, paired = d, j
            if paired is not None:
                if np.random.randint(2) == 0:
                    child_ids.append(idsA[i]); child_enh.append(enhA[i]); child_inh.append(inhA[i])
                    pA_count += 1
                else:
                    child_ids.append(idsB[paired]); child_enh.append(enhB[paired]); child_inh.append(inhB[paired])
                    pB_count += 1
                pB_range = list(set(pB_range) - {paired})
                pA_remaining = list(set(pA_remaining) - {i})

        # 3. Leftovers
        if (pA_count + pB_count) == 0:
            prob = 0.5
        else:
            prob = pA_count / (pA_count + pB_count)

        chosen_parent, chosen_range = (1, pA_remaining) if np.random.random() < prob else (2, pB_range)
        for idx in chosen_range:
            if chosen_parent == 1:
                child_ids.append(idsA[idx]); child_enh.append(enhA[idx]); child_inh.append(inhA[idx])
            else:
                child_ids.append(idsB[idx]); child_enh.append(enhB[idx]); child_inh.append(inhB[idx])

        # 4. Dynamics
        beta = betaA if np.random.random() < 0.5 else betaB
        delta = deltaA if np.random.random() < 0.5 else deltaB

        return encode_genome(beta, delta,
                             np.array(child_ids),
                             np.array(child_enh),
                             np.array(child_inh))

    # Make both children
    c1 = make_child(ids1, enh1, inh1, beta1, delta1,
                    ids2, enh2, inh2, beta2, delta2)
    c2 = make_child(ids2, enh2, inh2, beta2, delta2,
                    ids1, enh1, inh1, beta1, delta1)

    # Wrap in DEAP Individuals
    return creator.Individual(c1.tolist()), creator.Individual(c2.tolist())


def protein_distance_single(ids1, enh1, inh1, ids2, enh2, inh2,
                            i, j,
                            id_coef=1.0, inh_coef=1.0, enh_coef=1.0):
    """Compute distance between protein i from genome1 and protein j from genome2"""
    return (abs(ids1[i] - ids2[j]) * id_coef +
            abs(inh1[i] - inh2[j]) * inh_coef +
            abs(enh1[i] - enh2[j]) * enh_coef)