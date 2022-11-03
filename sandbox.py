import networkx as nx
import networkx.generators.random_graphs as random_graphs
import numpy as np
from scipy.stats import norm


import math

create_prob = 0.1
for i in range(100):
    if np.random.choice([True, False], p=[create_prob, 1-create_prob]):
        print(np.random.choice([True, False], p=[create_prob, 1-create_prob]))