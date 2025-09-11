import numpy as np
import time
from grn import GRN
from visulaizer import GRNVisualizer
from matplotlib import pyplot as plt
from genome import *

genes = random_genome(nin=1, nout=1, nreg=3)
g = GRN(genes, nin=1, nout=1)

# Example: drive GRN with sinusoidal input
g.reset()
g.warmup(25)

vis = GRNVisualizer(g)

timesteps = 100
for t in range(timesteps):
    # sinusoidal input between 0 and 1
    inp = 0.5 * (1 + np.sin(2 * np.pi * t / 20))  
    g.set_input([inp])   # assuming nin = 1
    g.step(1)
    
    vis.update()
    time.sleep(0.05)  # slow down so animation is visible