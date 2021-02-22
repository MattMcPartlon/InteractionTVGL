import os
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd

# target : [ty, name, m, M, pr, L, i, uid s1, s2, target_clust, final,AVG]
TY, NAME, m, M, PREC, SEQ_LEN, IDX, UID, S1, S2, TARGET_CLUST, FINAL, AVG = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12

# top scoring prediction rank by average precision for top n predicted contacts
# seps 5-9, 10-23, >23

# mean precision values for L, L/2, L/5, L/10
# sep  [i-j]>4, [i-j]>8, [i-j]>11, [i-j]>23

# same for criteria "any heavy atom <6A"

# correlation of precision (Top L, Top L/2) to alignment length (weighted and unweighted)

# average precision for different sparsities values use (0,0.02), (0.02,0.04),(0.04,0.10),(0.10,0.30),(>0.30)
from utils.utils import load_npy

meff_data = load_npy('/home/mmcpartlon/suppdata/meffs.npy')


def precision_data(entry):
    data = defaultdict(list)
    if entry[PREC] is None:
        return None
    for k, v in entry[PREC].items():
        data[entry[m]].append(v)
    return data


def filter_first_clust(entry, min_sparsity=0, max_sparsity=1, *args, **kwargs):
    return (min_sparsity <= entry[S2] <= max_sparsity) and entry[IDX] == 0


def filter_target_clust(entry, min_sparsity=0, max_sparsity=1, *args, **kwargs):
    return (min_sparsity <= entry[S2] <= max_sparsity) and entry[TARGET_CLUST]


def filter_not_first_or_target(entry, min_sparsity=0, max_sparsity=1, *args, **kwargs):
    t1 = filter_first_clust(entry, min_sparsity, max_sparsity)
    t2 = filter_target_clust(entry, min_sparsity, max_sparsity)
    return (not t1) and (not t2) and not entry[AVG]

def filter_avg(entry):
    return entry[AVG]


def filter_top(entry, ty='TOP_L', *args, **kwargs):
    return entry[TY] == ty


def get_data(data, data_func, filter_func):
    all_data = defaultdict(lambda: defaultdict(list))
    for name in data:
        target_name = name.split('_')[0]
        if target_name.endswith('.npy'):
            target_name = target_name[:-4]
        target_dat = data[name]
        for entry in target_dat:
            #print(entry[FINAL])
            if filter_func(entry):
                to_add = data_func(entry)
                if to_add is None:
                    continue
                for k, v in to_add.items():
                    all_data[target_name][k].append(v)
    return all_data


def get_quantiles_n_avgs(data, qs=(0.25, 0.5, 0.75, 1)):
    stat_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name in data:
        target_dat = data[name]
        if name.endswith('.npy'):
            name = name[:-4]
        for k, v in target_dat.items():
            if len(v) == 0:
                continue
            q_vals = np.quantile(np.array(v), axis=0, q=qs)
            assert len(q_vals[0]) == len(v[0])
            for q, vals in zip(qs, q_vals):
                stat_data[name][k][q] = vals
            stat_data[name][k]['avg'] = np.mean(np.array(v), axis=0)
    return stat_data


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


def get_msa_depth(target_msa_path):
    count = 0
    with open(target_msa_path, 'r+') as f:
        for x in f:
            if x.startswith('>'):
                count += 1
    return count


def get_meff(target_name):
    return meff_data[target_name]


# table
# for each type (quantile, avg):
# method, target, precision (1-50), precision (top L), (min_sep)
# target clust, first clust, all, other clust
# sparsity


data_root = sys.argv[1]
seq_root = sys.argv[2]
msa_root = sys.argv[3]
save_path = sys.argv[4]

seq_n_msa_info = {}
for ptn in os.listdir(seq_root):
    nm = ptn.split('.')[0]
    seq = get_seq(os.path.join(seq_root, ptn))
    msa_depth = get_msa_depth(os.path.join(msa_root, nm + '.aln.fasta'))
    seq_n_msa_info[nm] = {'len': len(seq), 'depth': msa_depth, 'meff': meff_data[nm]}

methods = [x.split('.')[0] for x in os.listdir(data_root) if x.endswith('.npy')]
methods = [x for x in methods if '_top' not in x]
print(methods)
_header = 'method,target,seq len, msa depth, meff, cluster type,s1,s2, min sep, max sep, data type'
_header = [x.strip() for x in _header.split(',')]

types = ['ALL', 'FIRST', 'TARGET', 'NOT FIRST OR TARGET', 'AVG']
data_funcs = [precision_data] * 4
filter_funcs = [
    lambda *args, **kwargs: True,
    filter_first_clust,
    filter_target_clust,
    filter_not_first_or_target,
    filter_avg,
]
finals = [True,True,False,False]
tys = ['TOP_L', 'NORMAL','TOP_L','NORMAL']
data_map = {}
for final,ty in zip(finals,tys):
    all_data = []
    prs = [1, 2, 5, 10, 25, 50, 100]
    if ty == 'TOP_L':
        prs = ['L', 'L/2', 'L/5', 'L/10']

    max_seps = {5: 9, 10: 23, 23: None}
    if ty == 'TOP_L':
        max_seps = {5: None, 9: None, 12: None, 24: None}
    header = list(_header)
    for pr in prs:
        header.append(str(pr))
    for method in methods:
        # get results for the method
        # upper and lower sparsity limits
        method_name = method
        data_path = os.path.join(data_root, method_name + '.npy')
        method_data = load_npy(data_path)
        print('processing method,', method)
        for type, dfunc, ffunc in zip(types, data_funcs, filter_funcs):
            fp = partial(ffunc, min_sparsity=0, max_sparsity=1)
            ft = partial(filter_top, ty=ty)
            ffunc = lambda entry, _method = method: (fp(entry) and ft(entry) and entry[FINAL]==final) or ('psicov' in _method and ft(entry))
            type_data = get_data(method_data, dfunc, ffunc)
            qs_and_avgs = get_quantiles_n_avgs(type_data)
            for target in qs_and_avgs:
                #print(target)
                #print(target in qs_and_avgs)
                t = target
                if target.endswith('npy'):
                    t = target[:-4]
                    exit(1)

                data_to_add = [method, t]
                data_to_add.append(seq_n_msa_info[t]['len'])
                data_to_add.append(seq_n_msa_info[t]['depth'])
                data_to_add.append(seq_n_msa_info[t]['meff'])
                data_to_add.extend([type, 0, 1])
                for min_sep in qs_and_avgs[target]:
                    _data_to_add = list(data_to_add)
                    _data_to_add.extend([min_sep, max_seps[min_sep]])
                    for data_type in qs_and_avgs[target][min_sep]:
                        __data_to_add = list(_data_to_add)
                        __data_to_add.append(data_type)
                        for v in qs_and_avgs[target][min_sep][data_type]:
                            __data_to_add.append(v)
                        all_data.append(__data_to_add)
    df = pd.DataFrame(all_data)
    print('data length',len(all_data))
    fl = '_final' if final else ''
    print(f'sheet name {ty}{fl}')
    data_map[f'{ty}_{fl}'] = (header,df)
with pd.ExcelWriter(save_path, mode='w+') as writer:
    for name,d in data_map.items():
        header, dat = d
        dat.to_excel(writer, sheet_name=name, header=header, float_format="%.5f")
