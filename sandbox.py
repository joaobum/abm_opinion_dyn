import networkx as nx
import networkx.generators.random_graphs as random_graphs
import numpy as np
from scipy.stats import norm


import multiprocessing


a = np.array([1, 1]).T

b = np.array([[1, 1],
              [-1, -1]])

cos = np.dot(a, b.T) / \
            (np.linalg.norm(a) * np.linalg.norm(b, axis=0))

cos = np.clip(cos, -1, 1)

a = np.array([[1, 1],
              [-1, -1],
              [1, 0]])


# print(np.dot(a[0], a[0]))
# print(np.linalg.norm(a, axis=1))
norms = np.linalg.norm(a, axis=1)
cos = np.dot(a, a.T) / np.outer(norms, norms)
print(np.degrees(np.arccos(cos)))




# print(np.dot(np.linalg.norm(a, axis=1).T, np.linalg.norm(a, axis=1)))
# cos = np.dot(a, b.T) / \
#             (np.linalg.norm(a) * np.linalg.norm(b, axis=0))
            
# print(np.arccos(cos))

# a = np.array([1, 2, 3])
# print(np.matmul(a.T, a))