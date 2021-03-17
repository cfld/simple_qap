# --

import sys
sys.path.append('build')

import networkx as nx
import numpy as np
from time import time

from simple_qap import qap
from scipy.optimize import quadratic_assignment

# --
# Make graphs

# np.random.seed(123)

n_nodes   = 256
n_seeds   = 0 # n_nodes // 10
p_edge    = 0.01

A = nx.adjacency_matrix(nx.erdos_renyi_graph(n_nodes, p=p_edge)).tocsr().A
# A = np.random.uniform(0, 100, (n_nodes, n_nodes))

p = np.arange(A.shape[0])
p[n_seeds:] = np.random.permutation(p[n_seeds:])
B = A[p][:,p]

# print(A.nnz, B.nnz)

A = np.ascontiguousarray(A).astype(np.int32)
B = np.ascontiguousarray(B).astype(np.int32)

# A_ = (exact_ppr(A, ppr_alpha) * 1000).astype(np.int32)
# B_ = (exact_ppr(B, ppr_alpha) * 1000).astype(np.int32)
# np.fill_diagonal(A_, 0) # !! Hack -- what to do?
# np.fill_diagonal(B_, 0) # !! Hack -- what to do?

null_score = (A * B).sum()
best_score = (A * A).sum()
print('best_score', best_score)

# --

t = time()
z = qap(A, B, verbose=1, piters=1, popsize=1)
sq_time = time() - t

sq_score = (A * B[:,z][z]).sum()

t = time()
res = quadratic_assignment(A, B, method='faq', options={"maximize" : True})
faq_score = (A * B[:,res.col_ind][res.col_ind]).sum()
faq_time  = time() - t

print(best_score, null_score, sq_score, faq_score, sq_time, faq_time)