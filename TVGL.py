import numpy as np
from InferenceGraph import TGraphVX
import builtins
from cvxpy import *


def TVGL(data, lengthOfSlice, lamb, beta, indexOfPenalty, verbose=False, eps=3e-3, epsAbs=1e-3, epsRel=1e-3):
    if indexOfPenalty == 1:
        print('Use l-1 penalty function')
    elif indexOfPenalty == 2:
        print('Use l-2 penalty function')
    elif indexOfPenalty == 3:
        print('Use laplacian penalty function')
    elif indexOfPenalty == 4:
        print('Use l-inf penalty function')
    else:
        print('Use perturbation node penalty function')
    gvx = TGraphVX(ty=indexOfPenalty)

    numberOfTotalSamples = data.shape[0]
    timestamps = int(numberOfTotalSamples / lengthOfSlice)
    size = data.shape[1]
    # Generate empirical covariance matrices
    sampleSet = []  # list of array
    k = 0

    empCovSet = []  # list of array


    for i in range(timestamps):
        # Generate the slice of samples for each timestamp from data
        k_next = builtins.min(k + lengthOfSlice, numberOfTotalSamples)
        samples = data[k: k_next, :]
        k = k_next
        sampleSet.append(samples)

        empCov = GenEmpCov(sampleSet[i].T)
        empCovSet.append(empCov)

    # delete: for checking
    print(f"Processing {len(sampleSet)} samples...")  #
    # print empCovSet
    print(('lambda = %s, beta = %s' % (lamb, beta)))
    print('cov shape :',empCovSet[0].shape)

    # Define a graph representation to solve
    for i in range(timestamps):
        n_id = i
        #
        theta_i = semidefinite(size, name='theta')
        # unclear why the L1 component is commented out
        obj = -log_det(theta_i) + trace(empCovSet[i] * theta_i)  # + alpha*norm(S,1)
        gvx.AddNode(n_id, obj)

        if i > 0:  # Add edge to previous timestamp
            currVar = gvx.GetNodeVariables(n_id)
            prevVar = gvx.GetNodeVariables(n_id - 1)
            edge_obj = beta * norm(currVar['theta'] - prevVar['theta'], indexOfPenalty)
            gvx.AddEdge(n_id, n_id - 1, Objective=edge_obj)

        # Add fake nodes, edges
        gvx.AddNode(n_id + timestamps)
        gvx.AddEdge(n_id, n_id + timestamps, Objective=lamb * norm(theta_i, 1))

    # need to write the parameters of ADMM
    gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose=verbose)

    # Extract the set of estimated theta
    thetaSet = []
    for nodeID in range(timestamps):
        val = gvx.GetNodeValue(nodeID, 'theta')
        thetaEst = upper2FullTVGL(val, eps)
        thetaSet.append(thetaEst)
    return thetaSet


def GenEmpCov(samples, useKnownMean=False, m=0):
    # samples should be array
    size, samplesPerStep = samples.shape
    if not useKnownMean:
        m = np.mean(samples, axis=1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:, i]
        empCov = empCov + np.outer(sample - m, sample - m)
    empCov = empCov / samplesPerStep
    return empCov


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
