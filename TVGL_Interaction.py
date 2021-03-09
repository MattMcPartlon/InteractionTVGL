import importlib
from typing import List
from typing import NamedTuple
import logging

import numpy as np
from cvxpy import *

import InferenceGraph
from InferenceGraph import TGraphVX

logger = logging.getLogger(__name__)

class PenaltyType(NamedTuple):
    L1 = 1
    L2 = 2
    LAPLACIAN = 3
    LINF = 4
    PERTURBATION = 5

def wrap_list(v, size):
    if isinstance(v, List) or isinstance(v, np.ndarray):
        ret = v
    elif not (isinstance(v,float) or isinstance(v,int)):
        raise Exception(f'unexpected value for v: {v}')
    else:
        ret = np.array([v]*size)
    if len(ret)!=size:
        raise Exception(f'got incorrect size {len(v)} for v = {v}, expected size {size}')
    return np.array(ret)



def TVGL(emp_cov_mats, n_processors, lamb, beta, indexOfPenalty,w,
         max_iters=200, verbose=False, epsAbs=1e-3,
         epsRel=1e-3):
    # reload inference graph - this is neccesary for multithreading+multiprocessing
    # used when solving ADMM. If this is not done, the global interpreter state
    # (which is copied at the time the multiprocessing pool is created) may copy
    # locks on threads (which for some reason are being held by the main process???)
    # causing deadlock.
    importlib.reload(InferenceGraph)
    if len(emp_cov_mats) == 0:
        return []
    lamb = wrap_list(lamb, len(emp_cov_mats))
    beta = wrap_list(beta, len(emp_cov_mats)-1)
    w = wrap_list(w, len(emp_cov_mats))

    gvx = TGraphVX(ty=indexOfPenalty)
    n_obs, size = len(emp_cov_mats), emp_cov_mats[0].shape[0]
    logger.info(f'n_obs : {n_obs}')
    logger.info(f'size : {size}')
    # Define a graph representation to solve
    logger.info('setting up solver graph')
    for i, mat in enumerate(emp_cov_mats):
        _lam = lamb[i]
        theta_i = semidefinite(size, name='theta')
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
    logger.info('finished setting up solver graph... ')
    # need to write the parameters of ADMM
    logger.info('solving... ')
    gvx.Solve(NumProcessors=n_processors, EpsAbs=epsAbs, EpsRel=epsRel,
              Verbose=verbose, MaxIters=max_iters, _node_weights=w)

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
