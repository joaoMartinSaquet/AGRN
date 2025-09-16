import sys
from pathlib import Path

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent.parent

# Add it to sys.path
sys.path.append(str(parent_dir))

import matplotlib.pyplot as plt
import numpy as np
from evolver import *
from genome import *
from problem import *
from grn import *


if __name__ == "__main__":

    p = gymProblem("MountainCarContinuous-v0", start_nreg=0)
    
    e = EATMuPlusLambda(nin=p.nin, nout=p.nout, nreg=0)
    alg = e.run(5, p, 100, 100, multiproc=True, verbose=True)
    e.visualize_evolutions()
    p.vis_genome(alg[0][0])
    
    print("best len : ", len(alg[0][0]))
    print("best : ", alg[0][0])
    plt.show()