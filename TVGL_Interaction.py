import importlib
from typing import List
from typing import NamedTuple

import numpy as np
from cvxpy import *

import InferenceGraph
from InferenceGraph import TGraphVX


class PenaltyType(NamedTuple):
    L1 = 1
    L2 = 2
    LAPLACIAN = 3
    LINF = 4
    PERTURBATION = 5


def print_message(msg, verbose=True):
    if verbose:
        print(msg + '\n')


def TVGL(emp_cov_mats, n_processors, lamb, beta, indexOfPenalty,
         max_iters=200, verbose=False, epsAbs=1e-3,
         epsRel=1e-3, use_cluster=False):
    importlib.reload(InferenceGraph)
    if len(emp_cov_mats) == 0:
        return []
    if isinstance(lamb, List) or isinstance(lamb, np.ndarray):
        if len(lamb) != len(emp_cov_mats):
            print(f'got lam : {lamb}, but {len(emp_cov_mats)} cov mats')
            assert False
    else:
        lamb = [lamb] * len(emp_cov_mats)
    if isinstance(beta, List) or isinstance(beta, np.ndarray):
        assert len(beta) == len(emp_cov_mats) - 1
    else:
        beta = [beta] * (len(emp_cov_mats) - 1)

    gvx = TGraphVX(ty=indexOfPenalty)
    n_obs, size = len(emp_cov_mats), emp_cov_mats[0].shape[0]
    print('n_obs', n_obs)
    print('size', size)
    print((size ** 2) * n_obs)
    # Define a graph representation to solve
    print_message('setting up solver graph...')
    for i, mat in enumerate(emp_cov_mats):
        _lam = lamb[i]
        theta_i = semidefinite(size, name='theta')
        # unclear why the L1 component is commented out
        obj = -log_det(theta_i) + trace(mat * theta_i)  # + alpha*norm(S,1)
        gvx.AddNode(i, obj)
        if i > 0:  # Add edge to previous timestamp
            _beta = beta[i - 1]
            currVar = gvx.GetNodeVariables(i)
            prevVar = gvx.GetNodeVariables(i - 1)
            edge_obj = _beta * norm(currVar['theta'] - prevVar['theta'], indexOfPenalty)
            gvx.AddEdge(i, i - 1, Objective=edge_obj)

        # Add fake nodes, edges
        gvx.AddNode(i + n_obs)
        gvx.AddEdge(i, i + n_obs, Objective=_lam * norm(theta_i, 1))
    print_message('finished setting up solver graph... ', verbose)
    # need to write the parameters of ADMM
    print_message('solving... ', verbose)
    gvx.Solve(NumProcessors=n_processors, EpsAbs=epsAbs, EpsRel=epsRel,
              Verbose=verbose, MaxIters=max_iters, UseClustering=use_cluster)

    # Extract the set of estimated theta
    thetaSet = []
    for nodeID in range(n_obs):
        val = gvx.GetNodeValue(nodeID, 'theta')
        thetaEst = upper2FullTVGL(val, eps=0)
        thetaSet.append(thetaEst)
    return thetaSet


def upper2FullTVGL(a: np.ndarray, eps: float = 0) -> np.ndarray:
    # a should be array
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    d = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(d))
    return A
