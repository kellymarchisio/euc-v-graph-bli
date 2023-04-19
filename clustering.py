import numpy as np
import tensorflow_datasets as tfds
from collections import defaultdict

def cluster_graph(directed_adj_matrix: np.array, theta: float, k: int, dictionary_form: bool = True) -> dict:
	n = len(directed_adj_matrix)
	clusters = defaultdict(lambda: defaultdict(int))
	if k <= 0:
	  return
	for m in range(1,k+1):
	  matrix = np.linalg.matrix_power(directed_adj_matrix, m)
	  for i in range(n):
	    for j in range(n):
	      if matrix[i][j] > theta:
	        clusters[i][j] = 1
	        print(clusters)
	for i in clusters.keys():
         clusters[i] = dict(clusters[i])
	clusters = dict(clusters)
	if not dictionary_form:
	  for i in range(n):
           clusters[i] = clusters[i].keys()
	return clusters

mat = np.array([[1, 0.5],[0.3, 1]])
theta = 0.4
k = 1
print(cluster_graph(mat, theta, k))
