import math

import numpy as np
import multiprocessing

# Returns True if convergence criteria have been satisfied
# eps_abs = eps_rel = 0.01
# r = Ax - z
# s = rho * (A^T)(z - z_old)
# e_pri = sqrt(p) * e_abs + e_rel * max(||Ax||, ||z||)
# e_dual = sqrt(n) * e_abs + e_rel * ||rho * (A^T)u||
# Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
# Returns (boolean shouldStop, primal residual value, primal threshold,
#          dual residual value, dual threshold)
#x, z, z_old, u, rho, p, n, e_abs, e_rel, verbose = [None]*10
def init(x_, z_, z_old_, u_):
    global x, z, z_old, u, rho, p, n, e_abs, e_rel, verbose
    x, z, z_old, u

def write_a_results(i,*data):
    global a_results
    a_results[i] = data

def write_a_tr_results(i,*data):
    global a_tr_results
    a_tr_results[i] = data


def A_norms(x, z, rc_tups_row_sorted):
    Ax = numpy.zeros(len(rc_tups_row_sorted))
    # assert len(Ax) == rc_tups_row_sorted[-1][0] + 1
    # assert len(Ax) == len(x) == len(z)
    mn, mx = rc_tups_row_sorted[0][0], rc_tups_row_sorted[-1][0]
    for i, j in rc_tups_row_sorted:
        Ax[i - mn] = x[j]
    res_pri_sq = numpy.sum(numpy.square(Ax - z[mn:mx + 1]))
    norm_AX_sq = numpy.sum(numpy.square(Ax))
    norm_z_sq = numpy.sum(numpy.square(z[mn:mx + 1]))
    return res_pri_sq, norm_AX_sq, norm_z_sq


def A_tr_norms(z, z_old, u, rho, rc_tups_col_sorted):
    col_norm_buffer = numpy.zeros(rc_tups_col_sorted[-1][1] + 1)
    offset = rc_tups_col_sorted[0][1]
    # calculate dual residual
    z_diff = z - z_old
    col_norm_buffer[:] = 0

    for i, j in rc_tups_col_sorted:
        col_norm_buffer[j - offset] += z_diff[i]
    res_dual = numpy.sum(numpy.square(rho * col_norm_buffer))

    # calculate A_tr dot u norm
    col_norm_buffer[:] = 0
    for i, j in rc_tups_col_sorted:
        col_norm_buffer[j - offset] += u[i]
    Atr_u_norm = numpy.sum(numpy.square(rho * col_norm_buffer))
    return res_dual, Atr_u_norm


def driver(x, z, z_old, u, rho, p, n,
           e_abs, e_rel, verbose, rc_tups_col_sorted, rc_tups_row_sorted, chunks):
    # split input into chunks
    chunk_len = len(rc_tups_row_sorted) // chunks
    # res_pri_sq, norm_AX_sq, norm_z_sq

    A_data = numpy.zeros((chunks, 3))
    for i in range(chunks):
        s, e = chunk_len * i, chunk_len * (i + 1)
        if i == chunks - 1:
            e = len(rc_tups_row_sorted)
        A_data[i, :] = A_norms(x, z, rc_tups_row_sorted[s:e])

    # same thing for A_tr
    # res_dual, Atr_u_norm
    Atr_data = numpy.zeros((chunks, 2))
    chunk_start = 0
    for i in range(chunks):
        s, e = chunk_start, chunk_len * (i + 1)
        if i == chunks - 1:
            e = len(rc_tups_row_sorted)
        while e < len(rc_tups_row_sorted) and rc_tups_col_sorted[e][1] == rc_tups_col_sorted[e - 1][1]:
            e += 1
        Atr_data[i, :] = A_tr_norms(z, z_old, u, rho, rc_tups_col_sorted[s:e])
        chunk_start = e

    x = numpy.sqrt(numpy.sum(A_data, axis=0))
    y = numpy.sqrt(numpy.sum(Atr_data, axis=0))
    res_pri, norm_Ax, norm_z = x
    res_dual, Atr_u_norm = y
    # Primal and dual thresholds. Add .0001 to prevent the case of 0.
    e_pri = math.sqrt(p) * e_abs + e_rel * max(norm_Ax, norm_z) + .0001
    e_dual = math.sqrt(n) * e_abs + e_rel * Atr_u_norm + .0001
    # Primal and dual residuals

    if verbose:
        # Debugging information to print convergence criteria values
        print('  res_pri:', res_pri)
        print('  e_pri:', e_pri)
        print('  res_dual:', res_dual)
        print('  e_dual:', e_dual)
    stop = (res_pri <= e_pri) and (res_dual <= e_dual)
    return (stop, res_pri, e_pri, res_dual, e_dual)





