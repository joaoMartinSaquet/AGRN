# AGRN: Artificial Gene Regulatory Networks

AGRN (Artificial Gene Regulatory Networks) is a Python framework for evolving and simulating gene regulatory networks using evolutionary algorithms. This project implements a sophisticated genetic algorithm system that can evolve GRNs to solve various computational problems, from regression tasks to reinforcement learning environments.

## Overview

Gene Regulatory Networks (GRNs) are biological networks that control gene expression patterns through regulatory relationships between genes. This project provides an artificial implementation that:

- Simulates protein interactions through enhancing and inhibiting factors
- Uses evolutionary algorithms to optimize network topology and parameters
- Supports both regression and reinforcement learning applications
- Includes visualization tools for network analysis

## Features

- **Dynamic Network Evolution**: Networks can add/remove regulatory proteins during evolution
- **Multiple Problem Types**: Support for regression problems, pattern formation (French flag), and reinforcement learning tasks
- **Advanced Genetic Operators**: Custom crossover and mutation operators designed for GRN genomes
- **Real-time Visualization**: Interactive network visualization showing protein concentrations and interactions
- **Configurable Parameters**: YAML-based configuration for all evolutionary and network parameters
- **Performance Optimized**: Uses Numba JIT compilation for fast network simulation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AGRN

# Install dependencies
pip install -r requirements.txt  # Create this based on imports
```

Required dependencies:
- numpy
- matplotlib
- deap (Distributed Evolutionary Algorithms in Python)
- numba
- loguru
- gymnasium (for RL environments)
- networkx (for visualization)
- seaborn
- PyYAML

### Basic Usage

#### 1. Regression Example

```python
from agrn.evolver import EATMuPlusLambda
from agrn.problem import RegressionProblem
import numpy as np

# Create training data
t = np.linspace(0, 1, 500)
y = np.sin(t * 10) ** 2  # Target function

# Set up problem and evolver
problem = RegressionProblem(t, y, nin=1, nout=1, nreg=0)
evolver = EATMuPlusLambda(nin=1, nout=1, nreg=0)

# Evolve network
hof = evolver.run(n_gen=1000, problem=problem, mu=100, lambda_=300)

# Visualize results
evolver.visualize_evolutions()
```

#### 2. Reinforcement Learning Example

```python
from agrn.problem import gymProblem

# Set up RL problem
env_problem = gymProblem("MountainCarContinuous-v0", start_nreg=0)
evolver = EATMuPlusLambda(nin=env_problem.nin, nout=env_problem.nout, nreg=0)

# Evolve network
hof = evolver.run(n_gen=200, problem=env_problem, mu=500, lambda_=500)

# Test evolved network
env_problem.vis_genome(hof[0][0])
```

#### 3. Network Visualization

```python
from agrn.grn import GRN
from agrn.visulaizer import GRNVisualizer
from agrn.genome import random_genome

# Create random network
genome = random_genome(nin=1, nout=1, nreg=3)
grn = GRN(genome, nin=1, nout=1)

# Visualize network dynamics
vis = GRNVisualizer(grn)
for t in range(100):
    inp = 0.5 * (1 + np.sin(2 * np.pi * t / 20))
    grn.set_input([inp])
    grn.step(1)
    vis.update()
```

## Architecture

### Core Components

1. **GRN (`grn.py`)**: Core gene regulatory network simulation
   - Protein concentration dynamics
   - Enhancing/inhibiting interactions
   - Numba-optimized computation

2. **Genome (`genome.py`)**: Genome encoding and manipulation
   - Flat genome representation: [beta, delta, identifiers, enhancers, inhibiters]
   - Encoding/decoding functions
   - Distance metrics for speciation

3. **Evolver (`evolver.py`)**: Evolutionary algorithm implementation
   - μ+λ evolution strategy
   - Custom genetic operators
   - Statistics tracking and visualization

4. **Genetic Operators**:
   - **Crossover (`crossover.py`)**: Protein-aware crossover with similarity matching
   - **Mutation (`mutation.py`)**: Add/delete/modify mutations for network topology

5. **Problems (`problem.py`)**: Problem definitions
   - `RegressionProblem`: Function approximation
   - `FrenchFlagProblem`: Pattern formation task
   - `gymProblem`: Reinforcement learning environments

6. **Visualization (`visulaizer.py`)**: Real-time network visualization

### Configuration

The `config.yaml` file contains all evolutionary parameters:

```yaml
POPULATION_SIZE: 100
INITIALIZATION_DUPLICATION: 10
TOURNAMENT_SIZE: 3
CROSSOVER_RATE: 0.25
MUTATION_RATE: 0.75
START_REGULATORY_SIZE: 1
# ... more parameters
```

## Network Dynamics

The GRN simulation follows these dynamics:

1. **Protein Interactions**: Each protein has identifier, enhancer, and inhibitor values
2. **Affinity Calculation**: Interaction strength based on protein similarity
3. **Concentration Updates**: Protein concentrations evolve based on:
   ```
   dc_i/dt = δ * (enhancing_factor - inhibiting_factor)
   ```
4. **Input/Output**: Input proteins have fixed concentrations, output proteins are read

## Examples

See the `example/` directory for complete examples:

- `regressions.py`: Function approximation with Fourier series
- `reinforcement.py`: OpenAI Gym environment control

## File Structure

```
AGRN/
├── agrn/                    # Core package
│   ├── grn.py              # Network simulation
│   ├── genome.py           # Genome representation
│   ├── evolver.py          # Evolution engine
│   ├── problem.py          # Problem definitions
│   ├── crossover.py        # Crossover operators
│   ├── mutation.py         # Mutation operators
│   ├── visulaizer.py       # Visualization
│   └── test/               # Tests and examples
├── example/                # Usage examples
│   ├── regressions.py      # Function approximation
│   └── reinforcement.py    # RL examples
├── out/                    # Output directory
├── config.yaml             # Configuration file
└── README.md              # This file
```

## Advanced Usage

### Custom Problems

Implement the problem interface:

```python
class CustomProblem:
    def __init__(self, nin, nout, nreg):
        self.nin = nin
        self.nout = nout
        self.nreg = nreg
    
    def eval(self, genome):
        # Create GRN from genome
        grn = GRN(genome, self.nin, self.nout)
        
        # Evaluate fitness
        fitness = your_evaluation_function(grn)
        
        return fitness,
```

### Parameter Tuning

Key parameters to tune:

- `BETA_MIN/MAX`: Affinity calculation range
- `DELTA_MIN/MAX`: Dynamics speed
- `MUTATION_RATE`: Balance exploration vs exploitation
- `CROSSOVER_THRESHOLD`: Protein similarity for crossover
- `START_REGULATORY_SIZE`: Initial network complexity

## Research Applications

This framework has been used for:

- **Function Approximation**: Learning complex mathematical functions
- **Pattern Formation**: Recreating biological pattern formation processes
- **Control Problems**: Evolving controllers for dynamic systems
- **Artificial Life**: Studying emergent behaviors in regulatory networks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Specify your license here]

## Citations

If you use this code in your research, please cite:

```
[Add appropriate citation information]
```

## Acknowledgments

- Built using the DEAP evolutionary algorithms framework
- Inspired by biological gene regulatory network research
- Thanks to the open-source Python scientific computing community
