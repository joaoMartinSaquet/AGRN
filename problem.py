import grn
import numpy as np
import loguru as logger
import gymnasium as gym


class RegressionProblem:
    def __init__(self,input_features, output_features, nin, nout, nreg):
        self.xtrain = input_features
        self.ytrain = output_features

        self.nin = nin
        self.nout = nout
        self.nreg = nreg



    def eval(self, genome):
        # translate the model from the genome
        
        g = grn.GRN(genome)

        g.setup()
        ypred = self.run_grn(g)        
        err = np.sum(10*abs(ypred-self.ytrain)**2)
        if np.isnan(err):
            logger.warning("err is nan") 

        # print("error on problem", err)
        return -err, ypred
    
    def run_grn(self, grn):

        grn.reset()
        grn.warmup(25)
        ypred = []
        N = len(self.xtrain)

        for i in range(N):
            grn.set_input(self.xtrain[i])
            grn.step(100)
            ypred.append(grn.get_output().item())
            
        return ypred
    

class FrenchFlagProblem:
    def __init__(self, nin, nout, nreg):
        self.N = 30
        self.nin = 2
        self.nout = 3
        self.nreg = 0

        self.ytrain = self.french_flag(self.N)  

    def eval(self, genome):
        # translate the model from the genome
        
        g = grn.GRN(genome, self.nin, self.nout)

        g.setup()
        ypred = self.run_grn(g)        
        err = np.linalg.norm(ypred.T-self.ytrain)
        if np.isnan(err):
            logger.warning("err is nan") 

        # print("error on problem", err)
        return -err, ypred

    def run_grn(self, g):

        g.reset()
        g.warmup(25)
        ypred = np.zeros((self.N,self.N,3))

        for i in range(self.N):
            for j in range(self.N):
                g.set_input([i/self.N, j/self.N])
                g.step(10)
                out = g.get_output()
                ypred[i][j] = out
            
        return ypred
    
    def french_flag(self, N):
        """
        Create a NxNx3 matrix representing the French flag.
        
        Blue | White | Red
        """
        flag = np.zeros((N, N, 3), dtype=np.uint8)  # RGB uint8 values 0-255

        # Width of each stripe
        stripe_width = N // 3

        # Blue stripe
        flag[:, :stripe_width, :] = [0, 0, 255]  # RGB for blue

        # White stripe
        flag[:, stripe_width:2*stripe_width, :] = [255, 255, 255]

        # Red stripe
        flag[:, 2*stripe_width:, :] = [255, 0, 0]  # RGB for red

        return flag.T/255.0



class gymProblem():
    def __init__(self, env_name, start_nreg):
        self.env = gym.make(env_name)

        self.nin = self.env.observation_space.shape[0]
        self.nout = self.env.action_space.shape[0]
        self.nreg = start_nreg

        
        self.has_continuous_observation = isinstance(self.env.observation_space, gym.spaces.Box)
        
        action_space = self.env.action_space
        self.has_continuous_action = isinstance(action_space, gym.spaces.Box)
        self.dtype = float
        if self.has_continuous_action:
            self.nout = action_space.shape[0]
            self.h_act = action_space.high
            self.l_act = action_space.low
        else:
            self.nout = 1
            self.n = action_space.n
            self.dtype = int

    def eval(self, genome):
        
        g = grn.GRN(genome, self.nin, self.nout)
        obs, env_info = self.env.reset()
        done = False
        g.setup()
        g.warmup(25)
        
        ypred = self.run_grn(g)        
        err = np.linalg.norm(ypred.T-self.ytrain)
        if np.isnan(err):
            logger.warning("err is nan") 

        # print("error on problem", err)
        return -err, ypred
