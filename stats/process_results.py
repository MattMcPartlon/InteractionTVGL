import numpy as np
from utils.utils import contact_norms, precision, get_sparsity
import os
from collections import defaultdict
import sys

def get_fams_and_names(output_root):
    fams,names = [],[]
    for x in os.listdir(output_root):
        tmp = x.split('_')[0]
        if tmp.endswith('.npy'):
            tmp=tmp[:-4]
        fams.append(tmp)
        names.append(x)
    return fams,names


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

def process_precision_result(idx):
    root = '/mnt/c/Users/mm851/Downloads/interaction_tvgl/'
    predicted = root + 'results_6_small_clust/output/'
    ground_truth = root + 'structures/ground_truth/'
    seq_rt = root+'suppdata/seq'
    save_f = './results_5_same_corr_mat.npy'
    if len(sys.argv)>1:
        predicted = sys.argv[1]
        ground_truth = sys.argv[2]
        seq_rt = sys.argv[3]

    fams,names = get_fams_and_names(predicted)
    print(len(names))
    precision_data = defaultdict(list)
    uid = 0
    fam,name = list(zip(fams,names))[idx]
    uid+=1
    print('#########processing :',name,'#########################')
    predicted_path = os.path.join(predicted,name)
    gt_path = os.path.join(ground_truth,fam+'.native.npy')
    gt_data = np.load(gt_path, allow_pickle=True).item()
    ds =gt_data['atomDistMatrix']['CbCb']
    predicted_data = np.load(predicted_path, allow_pickle=True).item()
    if 'beta' in predicted_data:
        lam,beta = predicted_data['lam'],predicted_data['beta']
    else:
        lam, beta = 2.5, 12

    corr_mats = predicted_data['theta set']
    seq_path = os.path.join(seq_rt,fam+'.fasta')
    print('============================')
    print('target :',fam)
    print('lambda :',lam,'beta :',beta)
    print('num corr_mats :',len(corr_mats))
    print('seq length', len(get_seq(seq_path)))
    ps = []
    if corr_mats is None:
        return
    if len(corr_mats)==0:
        return
    if not isinstance(corr_mats[0],np.ndarray):
        return
    for j,theta in enumerate(corr_mats):
        #print('num non-zero elts',len(theta[theta!=0]))
        np.fill_diagonal(theta,0)
        nz = len(theta[theta!=0])
        #print('num non zero elements in theta ',nz)
        if nz == 0:
            #continue
            pass
        ps.append(contact_norms(theta))
        s1 = get_sparsity(theta)
        theta[np.abs(theta)<0.1*1e-3]=0
        print('sparsity theta:', get_sparsity(theta))
    for m,M in zip([5,10,23],[9,23,None]):
        for i,p in enumerate(ps):
            target_clust = False
            if 'clusters' in predicted_data:
                if get_seq(seq_path) in predicted_data['clusters'][i] and j > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            pr = precision(p, ds, min_sep=m-1, max_sep=M+1 if M is not None else M)
            print('theta idx :',i,'min sep :',m,'max_sep',M,'precision :',pr)
            print('sparsity ps', s2)
            print('----------------------------')
            print()
            precision_data[name].append([name,beta,lam,m,M,pr,len(get_seq(seq_path)),i, uid, name, s1, s2, target_clust])
    return precision_data


def process_precision_top_L_result(idx):
    predicted = sys.argv[1]
    ground_truth = sys.argv[2]
    seq_rt = sys.argv[3]
    fams,names = get_fams_and_names(predicted)
    print(len(names))
    precision_data = defaultdict(list)
    uid = 0
    fam,name = list(zip(fams,names))[idx]
    uid+=1
    print('#########processing :',name,'#########################')
    predicted_path = os.path.join(predicted,name)
    gt_path = os.path.join(ground_truth,fam+'.native.npy')
    gt_data = np.load(gt_path, allow_pickle=True).item()
    ds =gt_data['atomDistMatrix']['CbCb']
    predicted_data = np.load(predicted_path, allow_pickle=True).item()
    if 'beta' in predicted_data:
        lam,beta = predicted_data['lam'],predicted_data['beta']
    else:
        lam, beta = 2.5, 12

    corr_mats = predicted_data['theta set']
    seq_path = os.path.join(seq_rt,fam+'.fasta')
    print('============================')
    print('target :',fam)
    print('lambda :',lam,'beta :',beta)
    print('num corr_mats :',len(corr_mats))
    print('seq length', len(get_seq(seq_path)))
    L = len(get_seq(seq_path))
    ps = []
    if corr_mats is None:
        return
    if len(corr_mats)==0:
        return
    if not isinstance(corr_mats[0],np.ndarray):
        return
    for j,theta in enumerate(corr_mats):
        #print('num non-zero elts',len(theta[theta!=0]))
        np.fill_diagonal(theta,0)
        nz = len(theta[theta!=0])
        #print('num non zero elements in theta ',nz)
        if nz == 0:
            #continue
            pass
        x = contact_norms(theta)
        ps.append(x)
        print(x.shape)
        s1 = get_sparsity(theta)
        theta[np.abs(theta)<0.5*1e-3]=0
        print('sparsity theta:', get_sparsity(theta))
    for m,M in zip([5,9,12,24],[None,None,None,None]):
        for i,p in enumerate(ps):
            target_clust = False
            if 'clusters' in predicted_data:
                if get_seq(seq_path) in predicted_data['clusters'][i] and i > 0:
                    target_clust = True
            s2 = get_sparsity(p, thresh=0.1)
            top = [L,L//2,L//5,L//10]
            pr = precision(p, ds, min_sep=m-1, max_sep=M+1 if M is not None else M,top=top)
            print('theta idx :',i,'min sep :',m,'max_sep',M,'precision :',pr)
            print('sparsity ps', s2)
            print('----------------------------')
            print()
            precision_data[name].append([name,beta,lam,m,M,pr,len(get_seq(seq_path)),i, uid, name, s1, s2, target_clust])
    return precision_data



from multiprocessing import Pool

n_files = len(os.listdir(sys.argv[1]))
n_workers = n_files//2
save_f = sys.argv[4]
ty = sys.argv[5]
print('save_f',save_f)
print('ty:',ty)
if not os.path.exists(save_f):
    with Pool(n_workers) as p:
        if ty != 'top':
            data = p.map(process_precision_result,[i for i in range(n_files)])
        else:
            data = p.map(process_precision_top_L_result,[i for i in range(n_files)])

    all_data = {}
    for dat in data:
        if dat is not None:
            for name in dat.keys():
                all_data[name]=dat[name]
    np.save(save_f,all_data)




