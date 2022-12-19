import networkx as nx
import networkx.generators.random_graphs as random_graphs
import numpy as np
from scipy.stats import norm
import multiprocessing
import os
from itertools import combinations
import glob

b = [2] * 5
a = [1, 1, 2, 1, 1, 2, 2]



op = np.array([-0.9, 0.2, -0.2, 0.9])

print(np.diff(op))