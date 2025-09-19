import numpy as np
import time
from agrn import GRN, GRNVisualizer, random_genome
# Simulate background updates
import threading

# ---- Use your GRNVisualizer ----
if __name__ == "__main__":

    nin = 1
    nout = 1
    nreg = 14
    
    rg = random_genome(nin, nout, nreg)
    grn = GRN(rg, nin, nout)
    viz = GRNVisualizer(grn, interval=500)  # refresh every 500ms
    

    def run_simulation():
        t = np.linspace(0, 1, 100)
        in_sig = np.sin(t)**2
        i = 0   
        while True:
            grn.set_input(in_sig[i%len(in_sig)])
            grn.step()
            time.sleep(0.5)
    
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()

    viz.show()
