import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
import logging
logger = logging.getLogger(__name__)
import warnings

withoutIndices = lambda m, ids: np.delete(np.delete(m, ids, axis=0), ids, axis=1)

def recover_mat(mat, size, removed_posns):
    new_mat = np.zeros((size, size))
    mat_i = -1
    for i in range(size):
        if i in removed_posns:
            continue
        else:
            mat_i +=1
        mat_j = -1
        for j in range(size):
            if j in removed_posns:
                continue
            else:
                mat_j+=1
                new_mat[i,j] = mat[mat_i,mat_j]
    return new_mat


def parse_sequence(seq_path):
    if not os.path.exists(seq_path):
        logger.error(f'path to sequence {seq_path} does not exist!')
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
        logger.error(f'data path {path} does not exist')
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


def adjust_lam_and_beta(theta_set, lam_, beta_, prev_sparsities, prev_lams, prev_betas, target_sparsity, thresh=1e-3, sep=3, single_lam = False, idx = 1):
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
        sparsity_diff = (target_sparsity - sp) #/ target_sparsity
        if abs(sparsity_diff) < 0.2*1e-2 and sparsity_diff < 0:
            continue
        elif len(prev_lams) <=2 or len(prev_sparsities)%2==0:
            mx = min(4,abs(sp-target_sparsity)/target_sparsity)
            scale = np.random.uniform(1,1+mx)
            lam[i] *= scale if sparsity_diff < 0 else 1 / scale
            opt = False
        else:
            opt = False
    if not opt:
        if len(prev_sparsities) >2 and len(prev_sparsities)%2==1:
            logger.info('using gpr inverse to update lambda...')
            ps, pls, bs = list(prev_sparsities), list(prev_lams), list(prev_betas)
            ps.append(sparsities), pls.append(lam), bs.append(beta)
            if single_lam:
                pls = [[p[idx]] for p in pls]
                ps = [[p[idx]] for p in ps]
            lam = gpr_inverse(pls, ps, target_sparsity)

    if single_lam:
        if len(lam)==1:
            l = lam[0]
        else:
            l = lam[idx]
        lam = np.array([l]* len(lam_))

    return lam, beta, sparsities


def constant_ratio_beta_context(xs, ratio = 1/4):
    tmp = np.array(xs)
    if len(tmp.shape) == 1:
        tmp = np.array(xs).reshape(1,-1)
    fst = np.array(tmp[:,0])
    fst*=ratio
    cxt = np.hstack([fst for _ in range(len(tmp[0])-1)]).reshape(-1,len(tmp[0])-1)
    return np.concatenate((tmp,cxt),axis = 1)

from sklearn.gaussian_process.kernels import ConstantKernel,WhiteKernel,DotProduct

def get_kernel():
    k = ConstantKernel(constant_value_bounds=(1e-9, 1e9))*DotProduct(sigma_0_bounds=(1e-7,1e7))
    k+=WhiteKernel(noise_level_bounds=(1e-9,1e9))
    return k


def gpr_inverse(xs,ys,target,s=1e-7,e=10):
    try:
        with warnings.catch_warnings() as w:
            gpr = GaussianProcessRegressor(n_restarts_optimizer=5, copy_X_train=True, kernel=get_kernel())
            xs,ys = np.array(xs),np.array(ys)
            mx,my = np.mean(xs.ravel()),np.mean(ys.ravel())
            ys = ys/my
            xs = xs/mx
            gpr.fit(ys,xs)
            tgt = np.array([target/my]*len(ys[0]))
            lams = gpr.predict([tgt]).ravel()*mx
            try:
                logger.warning(str(w))
            except:
                pass
            return np.maximum(s,np.minimum(lams,e))
    except Exception as e:
        logger.error(f'error in gpr inv {e}')
        return np.array(xs[-1])*np.random.uniform(0.85,1.15,size=len(ys[0]))



def get_sparsity(arr, thresh=1e-3):
    temp = np.abs(np.copy(arr))
    temp[temp < thresh] = 0
    return len(temp[temp > 0]) / np.prod(temp.shape)


def average_product_correct(contact_norms):
    s_all = np.mean(contact_norms)
    s_cols = np.mean(contact_norms, axis=0)
    for i in range(len(contact_norms)):
        for j in range(len(contact_norms)):
            contact_norms[i, j] = contact_norms[i, j] -\
                                  ((s_cols[i] * s_cols[j]) / s_all)

from utils.bio_utils import AA_index_map
def contact_norms(arr, k=21):
    n = len(arr) // k
    assert len(arr)%k==0
    ns = np.zeros((n, n))
    for i in range(n):
        r = i * k
        for j in range(i+1,n):
            c = j * k
            #skip gap (always last letter ie index 20)
            for a in range(k-1):
                for b in range(k-1):
                    ns[i][j] += abs(arr[r + a][c + b])
    return ns+ns.T


def precision(predicted, native, min_sep=6, max_sep=None, top=None, cutoff = 8):
    if not top:
        top = [1, 2, 5, 10, 25, 50, 100]
    mat = np.copy(predicted)
    mat[np.tril_indices(len(mat), min_sep)] = 0
    max_sep = max_sep or len(mat)
    mat[np.triu_indices(len(mat),max_sep)]=0
    if len(mat[mat!=0])==0:
        return None
    native_contacts = np.zeros(native.shape)
    native_contacts[native <= cutoff] = 1
    precision = []
    for t in top:
        predicted_msk = get_contact_mask(mat, t)
        precision.append(np.sum(predicted_msk * native_contacts) / t)
    return {t:p for t,p in zip(top,precision)}


def get_contact_mask(cmat, n_contacts):
    temp = np.copy(cmat)
    temp = temp.flatten()
    idxs = np.argsort(-temp)
    temp[idxs[:n_contacts]]=1
    temp[idxs[n_contacts:]]=0
    return temp.reshape(cmat.shape)

def get_target_cluster(target_seq, all_seqs, thresh = 0.7):
    n = len(target_seq)
    cluster = []
    for seq in all_seqs:
        sim = sum([1 if t==o else 0 for t,o in zip(target_seq,seq)])/n
        if sim>thresh:
            cluster.append(seq)
    return cluster

def scale_spectrum(cov_mats):
    #trace = sum of eigenvalues    #all matrices are positive definite, so scaling
    #matrices to have same eigenvalue sum may make the
    #inverse cov matrix more similar?
    traces = np.array([np.trace(cov_mat) for cov_mat in cov_mats])
    max_trace = np.argmax(traces)
    scales = max_trace/traces
    return np.array([scale*c for scale,c in zip(scales,cov_mats)])


from scipy.optimize import minimize



def my_func(x,a,b,c,s):
    print(a*x-b,'ax-b')
    return s*np.log(abs(a*x-b))+c

def sp_func(pars, xs,ys):
    s = pars[1]
    a,b,c = pars[1:].reshape(3,-1)
    return np.linalg.norm(my_func(xs,a,b,c,s)-ys)**2

def fit_lam_inv(xs,ys,target_sp = 0.03, s=1e-7,e=1e7):
    x0 = np.ones(1+3*len(xs[0]))
    res = minimize(sp_func, x0, args = (ys,xs))
    x = res['x']
    s = x[0]
    a,b,c = x[1:].reshape(3,-1)
    tgt = np.array([target_sp]*len(ys[0]))
    return my_func(tgt,a,b,c,s).ravel()








