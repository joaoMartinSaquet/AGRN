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
    print("training on : ", env_name)
    p = gymProblem(env_name, start_nreg=0)
    
    e = EATMuPlusLambda(nin=p.nin, nout=p.nout, nreg=0)
    hof, hist = e.run(50, p.eval, 10, 200, multiproc=True, verbose=True, comma=True)
    e.visualize_evolutions()
    p.vis_genome(hof[0])
    
    print("best len : ", len(hof[0]))
    plt.show()