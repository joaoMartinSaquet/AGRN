from loguru import logger
import numpy as np


def compute_output_concentrations_diff(output_concentrations):
    lo = len(output_concentrations)
    if lo%2 != 0: logger.error("Number of outputs is not even")
    out = np.zeros((lo//2))
    j = 0
    for i in range(0, lo, 2):
        if  (output_concentrations[i] + output_concentrations[i+1]) == 0.0:
            o = 1.0
        else:
            # o = (output_concentrations[i] - output_concentrations[i+1]) / (output_concentrations[i] + output_concentrations[i+1])       
            o = (output_concentrations[i] + output_concentrations[i+1])/2  
        # out[j] = np.clip(o, 0.0, 1.0)
        j += 1

    return out
