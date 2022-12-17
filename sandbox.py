import networkx as nx
import networkx.generators.random_graphs as random_graphs
import numpy as np
from scipy.stats import norm
import multiprocessing
import os
from itertools import combinations
import glob

# social_graph = random_graphs.erdos_renyi_graph(
#             30, 0.3, seed=1)

# print(social_graph.degree[2])
# a = np.array([[1,2,3,4],
#               [1,2,3,4]])
# indices = [0, 2] 
# print(a[0][indices].sum()

file_list = glob.glob('/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data/*')
print(file_list)


