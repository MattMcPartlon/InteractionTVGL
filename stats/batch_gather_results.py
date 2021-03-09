import os
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np

from utils.utils import contact_norms, precision, get_sparsity
from typing import List

#target_list = '1a3aA	1a6mA	1aapA	1abaA	1atzA	1bebA	1behA	1brfA	1c44A	1c9oA	1cc8A	1chdA	1ctfA	1cxyA	1d1qA	1dbxA'
#target_set = set(target_list.split('\t'))

sparsity_cutoff = 1e-4

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
            if x.startswith('>'):
                continue
            else:
                target_seq = x.strip()
                break
    return target_seq


def process_precision_result(idx, results_root, ground_truth_root, seq_root, sp_cutoff = 1e-4, min_sep=1, project = False):
    global sparsity_cutoff
    sp_cutoff = sparsity_cutoff
    fams, names = get_fams_and_names(results_root)
    precision_data = defaultdict(list)
    uid = 0
    fam, name = list(zip(fams, names))[idx]
    #if fam not in target_list:
    #    return None
    print('#########processing :', name, '#########################')
    seq_path = os.path.join(seq_root, fam + '.fasta')
    results_path = os.path.join(results_root, name)
    ground_truth_path = os.path.join(ground_truth_root, fam + '.native.npy')
    gt_data = np.load(ground_truth_path, allow_pickle=True).item()
    ds = gt_data['atomDistMatrix']['CbCb']
    predicted_data = np.load(results_path, allow_pickle=True).item()
    precision_mats = predicted_data['theta set']
    if 'args' in predicted_data:
        if 'prec_thresh' in predicted_data['args']:
            sp_cutoff = predicted_data['args']['prec_thresh']
    print('sparsity cutoff',sp_cutoff)
    L = len(get_seq(seq_path))
    ps = []
    if precision_mats is None:
        return
    if len(precision_mats) == 0:
        return
    s1s = []
    if not isinstance(precision_mats,np.ndarray):
        precision_mats = np.array(precision_mats)
    if len(precision_mats.shape)==2:
        precision_mats=precision_mats.reshape((-1,precision_mats.shape[0],precision_mats.shape[1]))
    if precision_mats[0] is None:
        return
    l = len(precision_mats)
    sizes = [1]
    if 'clusters' in predicted_data:
        sizes = []
        clusters = predicted_data['clusters']
        if isinstance(clusters, List):
            clusters = {i:c for i,c in enumerate(clusters)}
        for c in clusters.values():
            sizes.append(len(c))
    wts = np.array(sizes)/np.sum(sizes)
    mean_prec = np.sum([x*w for x,w in zip(precision_mats,wts)],axis=0)
    precision_mats = [x for x in precision_mats]
    precision_mats.append(mean_prec)
    assert len(precision_mats)==l+1
    assert mean_prec.shape == precision_mats[0].shape
    for j, theta_ in enumerate(precision_mats):
        if theta_ is None:
            return

        #theta = project_ps(theta_)
        theta=theta_
        theta[np.abs(theta) < sp_cutoff] = 0
        nz = len(theta[theta != 0])
        if nz == 0:
            return

        vs = theta[np.triu_indices(len(theta),21*min_sep)]
        prec_arr = np.zeros(theta.shape)
        prec_arr[np.triu_indices(len(theta), 21 * min_sep)] = vs
        prec_arr+=prec_arr.T
        ps.append(contact_norms(prec_arr))
        s1s.append(get_sparsity(prec_arr))

    final = 'final' in name or 'clusters' not in predicted_data
    # compute precision for 5,10,23, sep
    for m, M in zip([5, 10, 23], [9, 23, None]):
        ty = 'NORMAL'
        for i, p in enumerate(ps):
            if project:
                p = project_ps(p)
            target_clust = False
            if 'clusters' in predicted_data and i!=len(ps)-1:
                if get_seq(seq_path) in predicted_data['clusters'][i] and i > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            pr = precision(p, ds, min_sep=m - 1, max_sep=M + 1 if M is not None else M)
            if i == len(ps)-1:
                precision_data[name].append([ty, name, m, M, pr, L, i, uid, name, s1s[i], s2, target_clust, final,True])
            else:
                precision_data[name].append([ty, name, m, M, pr, L, i, uid, name, s1s[i], s2, target_clust, final,False])


    for m, M in zip([5, 9, 12, 24], [None, None, None, None]):
        ty = 'TOP_L'
        for i, p in enumerate(ps):
            target_clust = False
            if 'clusters' in predicted_data and i!=len(ps)-1:
                if get_seq(seq_path) in predicted_data['clusters'][i] and i > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            top = [L, L // 2, L // 5, L // 10]
            pr = precision(p, ds, min_sep=m - 1, max_sep=M + 1 if M is not None else M, top=top)
            if i == len(ps) - 1:
                precision_data[name].append([ty, name, m, M, pr, L, i, uid, s1s[i], s2, target_clust, final,True])
            else:
                precision_data[name].append([ty, name, m, M, pr, L, i, uid, s1s[i], s2, target_clust, final,False])

    return precision_data


def project_ps(ps, dim = None):
    if dim is None:
        dim = int(len(ps)//3)
    dim = min(len(ps),dim)
    temp = np.copy(ps)
    temp[np.abs(ps)<1e-3]=0
    temp*=1e3
    d, q = np.linalg.eigh(temp)
    d[d<0]=0
    d/=1e3
    temp/=1e3
    cutoff = np.sort(np.abs(d))[-dim]
    d[np.abs(d)<cutoff]=0
    ps_ = np.dot(q*d,q.T)
    norms = [np.linalg.norm(ps),np.linalg.norm(temp[temp!=0]-ps_[temp!=0]),np.linalg.norm(temp)]
    print('norms (before, diff, temp) :',np.round(norms,5))
    return ps_


def process_results_root(results_root, ground_truth_root, seq_root, save_f, sp = 1e-4, project = False):
    global sparsity_cutoff
    sparsity_cutoff = sp
    n_files = len(os.listdir(results_root))
    n_workers = n_files#min(n_files, len(target_set))
    print('processing result root :',results_root)
    f = partial(process_precision_result, results_root=results_root, ground_truth_root=ground_truth_root,
                seq_root=seq_root, project = project)
    if not os.path.exists(save_f):
        with Pool(n_workers) as p:
            data = p.map(f, [i for i in range(n_files)])
        all_data = {}
        for dat in data:
            if dat is not None:
                for name in dat.keys():
                    all_data[name] = dat[name]
        np.save(save_f, all_data)


"""
Process results - for each data file, we write the precision information
of contact prediction obtained from the sample precision matrix.
"""
CUTOFF = 1
if __name__ == '__main__':
    data_root = sys.argv[1]  # pred
    ground_truth = sys.argv[2]
    seq_root = sys.argv[3]
    save_root = sys.argv[4]
    project = False
    print('data root',data_root)
    if len(sys.argv)>5:
        sparsity_cutoff = float(sys.argv[5])
    if len(sys.argv)>6:
        project = int(sys.argv[6])==1

    for data in os.listdir(data_root):
        #REMOVE LATER:
        if not ('remove_gaps' in data or 'psicov' in data):
            continue

        print('processing :', data)
        ext = ''
        if not os.path.isdir(os.path.join(data_root, data)):
            print('not a directory')
            continue
        if not (data.startswith('result') or data.startswith('target')):
            print('not a results folder')
            continue
        if 'output' not in os.listdir(os.path.join(data_root, data)):
            if 'results' not in os.listdir(os.path.join(data_root, data)):
                print('no results folder')
                continue
            else:
                ext = 'results'
        else:
            ext = 'output'

        data_dir = os.path.join(data_root, data, ext)
        if len(os.listdir(data_dir)) > CUTOFF:
            save_f = os.path.join(save_root, data + '.npy')
            process_results_root(results_root=data_dir, ground_truth_root=ground_truth, seq_root=seq_root,
                                 save_f=save_f,sp=sparsity_cutoff, project=project)
        else:
            print('too few results')
