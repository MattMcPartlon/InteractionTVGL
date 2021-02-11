import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor


def parse_sequence(seq_path):
    if not os.path.exists(seq_path):
        print(f'path to sequence {seq_path} does not exist!')
    with open(seq_path, 'r+') as f:
        for x in f:
            if x.startswith('>'):
                continue
            else:
                return x.strip()


def display_message(msg, priority=1, verbosity=1, verbose=1):
    verbose = verbosity or verbose
    if priority >= verbose:
        print(msg)


def load_npy(path):
    if not os.path.exists(path):
        print(f'data path {path} does not exist')
    data = np.load(path, allow_pickle=True)
    try:
        data = data.item()
    except:
        pass
    return data


def parse_seqs(aln_path):
    seqs = []
    with open(aln_path, 'r+') as f:
        for x in f:
            if not x.startswith('>') and len(x) > 1:
                seqs.append(x.strip())
    return seqs


def adjust_lam_and_beta(theta_set, lam_, beta_, prev_sparsities, prev_lams, target_sparsity, thresh=1e-3, sep=3):
    sparsities = []
    opt = True
    lam = list(lam_)
    beta = list(beta_)
    for i, t in enumerate(theta_set):
        theta = np.copy(t)
        # clear diagonal out to sep
        theta = theta[np.triu_indices(len(theta), 21 * sep)]
        # grab only the entries corresponding to separation > sep.
        sp = get_sparsity(theta, thresh=thresh)
        sparsities.append(sp)
        sparsity_diff = (target_sparsity - sp) / target_sparsity
        if abs(sparsity_diff) < 1e-2 and sparsity_diff < 0:
            continue
        elif len(prev_lams) < 2:
            lam[i] *= 2 if sparsity_diff < 0 else 1 / 2
            opt = False
        else:
            opt = False
    if not opt:
        if len(prev_sparsities) > 1:
            ps, pls = list(prev_sparsities), list(prev_lams)
            ps.append(sparsities), pls.append(lam)
            lam = gp_reg(pls, ps, target_sparsity)

    return lam, beta, sparsities


def gp_reg(xs, ys, target, s=0, e=1):
    gpr = GaussianProcessRegressor(n_restarts_optimizer=3)
    gpr.fit(xs, ys)
    # binary search for target sparsity
    n_samples = 30
    los, his = s * np.ones(len(xs[0])), e * np.ones(len(xs[0]))
    best_xs = np.array(xs[0])
    stop = False
    while not stop:
        test_xs = np.linspace(los, his, n_samples).reshape(n_samples, len(best_xs))
        pred = gpr.predict(test_xs)
        pred = np.abs(pred - target)
        best_idxs = np.argmin(pred, axis=0)
        assert len(best_idxs) == len(best_xs)
        best_xs = los + (his - los) * (best_idxs / (n_samples - 1))
        reduced_range = (his - los) * 2 / 5
        los, his = best_xs - reduced_range, best_xs + reduced_range
        stop = np.alltrue(np.abs(gpr.predict([best_xs]) - target) < 1e-4)
        stop = stop or np.alltrue(his[0] - los[0] < 1e-5)
    return best_xs


def get_sparsity(arr, thresh=1e-3):
    temp = np.abs(np.copy(arr))
    temp[temp < thresh] = 0
    return len(temp[temp > 0]) / np.prod(temp.shape)
