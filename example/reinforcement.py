import argparse
import matplotlib.pyplot as plt
import numpy as np
from agrn import EATMuPlusLambda, gymProblem


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='reinforcement',
                    description='Train a GRN model to solve a problem',)
    
    parser.add_argument('--env_name', type=str, default="MountainCarContinuous-v0")

    args = parser.parse_args()
    env_name = args.env_name    
    p = gymProblem(env_name, start_nreg=0)
    
    e = EATMuPlusLambda(nin=p.nin, nout=p.nout, nreg=0)
    alg = e.run(500, p, 500, 500, multiproc=True, verbose=True)
    e.visualize_evolutions()
    p.vis_genome(alg[0][0])
    
    print("best len : ", len(alg[0][0]))
    print("best : ", alg[0][0])
    plt.show()