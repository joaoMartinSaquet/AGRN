"""
AGRN: Artificial Gene Regulatory Networks

A Python framework for evolving and simulating gene regulatory networks
using evolutionary algorithms.
"""

from .grn import GRN
from .genome import random_genome, encode_genome, decode_genome, protein_distance
from .evolver import EATMuPlusLambda
from .problem import RegressionProblem, FrenchFlagProblem, gymProblem
from .mutation import mutate, add, delete, modify
from .crossover import cx
from .visulaizer import GRNVisualizer

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'GRN',
    'random_genome',
    'encode_genome', 
    'decode_genome',
    'protein_distance',
    'EATMuPlusLambda',
    'RegressionProblem',
    'FrenchFlagProblem', 
    'gymProblem',
    'mutate',
    'add',
    'delete',
    'modify',
    'cx',
    'GRNVisualizer'
]
