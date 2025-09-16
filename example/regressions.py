import matplotlib.pyplot as plt
import numpy as np
from agrn import EATMuPlusLambda, RegressionProblem, GRN

def f(t, f: float = 1, k: int = 3):
    """ Fourrier decomposition of a square signal

    Args:
        t (float): time
        f(float, optional): frequency. Defaults to 1. 
        k (int, optional): degree of decomposition. Defaults to 2.

    Returns:
        values
    """
    y = np.zeros(t.shape[0])
    for i in range(0, k):

        y += np.sin((2*i + 1) * 2*np.pi*f*t)/(2*i + 1)
    
    
    y /= (4/np.pi)
    
    # transform values between 0 and 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    # y = np.sin(t*10)**2
    return y

if __name__ == "__main__":

    
    t = np.linspace(0, 1, 500)
    ytrain = f(t, f=2, k=3)
    p = RegressionProblem(t, ytrain, nin=1, nout=1, nreg=0)


    e = EATMuPlusLambda(nin=1, nout=1, nreg=0)
    alg = e.run(1000, p, 100, 300, multiproc=True, verbose=True)
    e.visualize_evolutions()
    
    print("best len : ", len(alg[0][0]))
    print("best : ", alg[0][0])
    g = GRN(alg[0][0], nin=1, nout=1)
    p.run_grn(g)

    plt.plot(t, ytrain)
    plt.plot(t, p.run_grn(g))
    plt.show()