import networkx as nx
import networkx.generators.random_graphs as random_graphs
import numpy as np
from scipy.stats import norm

a = [1, 1, 1, 1, 1, 1, 1, 1, 0.5]

b = [1, 1, 1, 1, 1, 1, 1, 1, 1]

strength_ref = np.linalg.norm(a)/np.sqrt(len(a))

strength_ag = np.linalg.norm(b)/np.sqrt(len(b))

print(strength_ref/strength_ag)