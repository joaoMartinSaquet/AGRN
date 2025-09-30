import matplotlib.pyplot as plt
import numpy as np
from agrn import EATMuPlusLambda, RegressionProblem, GRN, FrenchFlagProblem
import networkx
# from deap import toolbox

def f(t, f: float = 1, k: int = 3):
    """ Fourrier decomposition of a square signal

    Args:
        t (float): time
        f(float, optional): frequency. Defaults to 1. 
        k (int, optional): degree of decomposition. Defaults to 2.
2025-09-24 16:45:53.167 | INFO     | agrn.evolver:run:184 - 194 |    -32.517 |    -37.889 |    -19.800 |    4.415 |    -33.712 |    50

    Returns:
        values
    """
    y = np.zeros(t.shape[0])
    for i in range(0, k):

        y += np.sin((2*i + 1) * 2*np.pi*f*t)/(2*i + 1)
    
    
    y /= (4/np.pi)
    
    # transform values between 0 and 1
    
    # y = np.sin(t*10)**2
    return y

if __name__ == "__main__":

    
    t = np.linspace(0, 1, 500)
    ytrain =  f(t, f=2, k=3)
    # ytrain = np.ones(t.shape[0])
    p = RegressionProblem(t, ytrain, nin=1, nout=1, nreg=0)


    e = EATMuPlusLambda(nin=p.nin, nout=p.nout, nreg=0)
    hof, hist = e.run(10, p.eval, 50, 1000, multiproc=True, verbose=True)


    # graph = networkx.DiGraph(hist.genealogy_tree)
    # graph = graph.reverse()     # Make the grah top-down
    # colors = [e.toolbox.evaluate(hist.genealogy_history[i])[0] for i in graph]
    # networkx.draw(graph, node_color=colors)
    # plt.show()


    e.visualize_evolutions()
    
    print("best len : ", len(hof[0]))
    print("best : ", hof[0])
    
    g = GRN(hof[0], p.nin, p.nout)
    yg = p.run_grn(g)

    print("np.sum(abs(yg - ytrain))", np.sum(abs(yg - ytrain)))
    # plt.plot(t, ytrain)
    plt.plot(t, yg)
    plt.plot(t, ytrain)
    plt.show()