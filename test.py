#!/usr/bin/env python

"""
  test.py
"""

import sys
sys.path.append('build')

import json
import numpy as np
from time import time
from io import StringIO

from simple_qap import qap
from scipy.optimize import quadratic_assignment

def read_prob(inpath):
  n = int(open(inpath).readlines()[0])
  
  x_str = ' '.join([xx.strip() for xx in open(inpath).readlines()[1:]])
  x     = np.loadtxt(StringIO(x_str))
  
  A = x[:n * n].reshape(n, n).astype(np.int32)
  B = x[n * n:].reshape(n, n).astype(np.int32)
  
  return A, B

# --
# IO

A, B = read_prob('data/qaplib/nug30.dat')

# --
# Run SQ

# !! runtime and solution quality increase w/ piters and popsize

t       = time()
res     = qap(A, B, piters=32, popsize=24, seed=123)
sq_time = time() - t

sq_score = (A * B[res][:,res]).sum()

# --
# Run Scipy baselines

t         = time()
faq_score = quadratic_assignment(A, B, method='faq').fun
faq_time  = time() - t

t         = time()
two_score = quadratic_assignment(A, B, method='2opt').fun
two_time  = time() - t

# --

print(json.dumps({
  "sq_time"   : float(sq_time),
  "faq_time"  : float(faq_time),
  "two_time"  : float(two_time),
  
  "sq_score"  : int(sq_score),
  "faq_score" : int(faq_score),
  "two_score" : int(two_score),
}))