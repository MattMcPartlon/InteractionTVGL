import os
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from utils.utils import contact_norms, precision, get_sparsity


def get_fams_and_names(output_root):
    fams, names = [], []
    for x in os.listdir(output_root):
        tmp = x.split('_')[0]
        if tmp.endswith('.npy'):
            tmp = tmp[:-4]
        fams.append(tmp)
        names.append(x)
    return fams, names


def get_seq(target_seq_path):
    target_seq = None
    assert os.path.exists(target_seq_path)
    with open(target_seq_path, 'r+') as f:
        for x in f:
            print(x)
            if x.startswith('>'):
                continue
            else:
                target_seq = x.strip()
                break
    return target_seq


def process_precision_result(idx, results_root=None, ground_truth_root=None, seq_root=None):
    fams, names = get_fams_and_names(results_root)
    precision_data = defaultdict(list)
    uid = 0
    fam, name = list(zip(fams, names))[idx]
    print('#########processing :', name, '#########################')
    seq_path = os.path.join(seq_root, fam + '.fasta')
    results_path = os.path.join(results_root, name)
    ground_truth_path = os.path.join(ground_truth_root, fam + '.native.npy')
    gt_data = np.load(ground_truth_path, allow_pickle=True).item()
    ds = gt_data['atomDistMatrix']['CbCb']
    predicted_data = np.load(results_path, allow_pickle=True).item()
    precision_mats = predicted_data['theta set']
    L = len(get_seq(seq_path))
    ps = []
    if precision_mats is None:
        return
    if len(precision_mats) == 0:
        return
    if not isinstance(precision_mats, np.ndarray):
        return
    for j, theta in enumerate(precision_mats):
        np.fill_diagonal(theta, 0)
        nz = len(theta[theta != 0])
        if nz == 0:
            return
        ps.append(contact_norms(theta))
        s1 = get_sparsity(theta)
        theta[np.abs(theta) < 0.1 * 1e-3] = 0

    final = 'final' in name or 'clusters' not in predicted_data
    # compute precision for 5,10,23, sep
    for m, M in zip([5, 10, 23], [9, 23, None]):
        ty = 'NORMAL'
        for i, p in enumerate(ps):
            target_clust = False
            if 'clusters' in predicted_data:
                if get_seq(seq_path) in predicted_data['clusters'][i] and i > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            pr = precision(p, ds, min_sep=m - 1, max_sep=M + 1 if M is not None else M)
            precision_data[name].append([ty, name, m, M, pr, L, i, uid, name, s1, s2, target_clust, final])

    for m, M in zip([5, 9, 12, 24], [None, None, None, None]):
        ty = 'TOP_L'
        for i, p in enumerate(ps):
            target_clust = False
            if 'clusters' in predicted_data:
                if get_seq(seq_path) in predicted_data['clusters'][i] and i > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            top = [L, L // 2, L // 5, L // 10]
            pr = precision(p, ds, min_sep=m - 1, max_sep=M + 1 if M is not None else M, top=top)
            precision_data[name].append([ty, name, m, M, pr, L, i, uid, name, s1, s2, target_clust, final])

    return precision_data


def process_results_root(results_root, ground_truth_root, seq_root, save_f):
    n_files = len(os.listdir(results_root))
    n_workers = n_files // 2
    f = partial(process_precision_result, results_root=results_root, ground_truth_root=ground_truth_root,
                seq_root=seq_root)
    if not os.path.exists(save_f):
        with Pool(n_workers) as p:
            data = p.map(f, [i for i in range(n_files)])
        all_data = {}
        for dat in data:
            if dat is not None:
                for name in dat.keys():
                    all_data[name] = dat[name]
        np.save(save_f, all_data)


CUTOFF = 15
if __name__ == '__main__':
    data_root = sys.argv[1]  # pred
    ground_truth = sys.argv[2]
    seq_root = sys.argv[3]
    save_root = sys.argv[4]

    for data in os.listdir(data_root):
        print('processing :', data)
        ext = ''
        if not os.path.isdir(os.path.join(data_root, data)):
            print('not a directory')
            continue
        if not (data.startswith('result') or data.startswith('target')):
            print('not a results folder')
            continue
        if 'output' not in os.listdir(os.path.join(data_root, data)):
            print('no output folder')
            continue
        else:
            ext = 'output'
        if 'results' not in os.listdir(os.path.join(data_root, data)):
            print('no results folder')
            continue
        else:
            ext = 'results'
        data_dir = os.path.join(data_root, data, ext)
        if len(os.listdir(data_dir)) > CUTOFF:
            save_f = os.path.join(save_root, data + '.npy')
            process_results_root(results_root=data_dir, ground_truth_root=ground_truth, seq_root=seq_root, save_f=save_f)
        else:
            print('too few results')
