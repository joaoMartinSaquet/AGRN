import grn
import numpy as np
import loguru as logger


class RegressionProblem:
    def __init__(self,input_features, output_features):
        self.xtrain = input_features
        self.ytrain = output_features

    def eval(self, genome):
        g = grn.GRN()
        g.from_genome(genome)

        ypred = self.run_grn(g)        
        err = np.linalg.norm(self.ytrain - np.array(ypred))
        if np.isnan(err):
            logger.warning("err is nan") 
        return -err
    
    def run_grn(self, grn):

        grn.reset()
        grn.step(25) # warmups
        ypred = []
        N = len(self.xtrain)

        for i in range(N):
            grn.set_input(self.xtrain[i])
            grn.step()
            ypred.append(grn.get_output())
            
        return ypred