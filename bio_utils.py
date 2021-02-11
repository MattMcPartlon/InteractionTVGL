from collections import defaultdict
from typing import List

import lazy_property
import numpy as np

from msa_utils import CalcSeqWeight

AA_alphabet = "ARNDCQEGHILKMFPSTWYV-"
AA_index_map = {aa: i for i, aa in enumerate(AA_alphabet)}
num_aas = len(AA_alphabet)
AA_vec_map = {}

for aa, i in AA_index_map.items():
    tmp = np.zeros(num_aas)
    tmp[i] = 1
    AA_vec_map[aa] = tmp


def check_seqs(seqs):
    if not seqs:
        assert False
    m = len(seqs[0])
    for seq in seqs:
        assert len(seq) == m


class Sequence:
    def __init__(self, sequence: str):
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'E')
        sequence = sequence.replace('X', '-')
        sequence = sequence.replace('U', '-')
        self.seq = sequence
        self.wt = None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]

    def set_wt(self, wt):
        self.wt = wt

    def get_wt(self):
        return self.wt

    def get_aa(self, pos):
        return self.seq[pos]

    def get_aa_vec(self, pos):
        v = self.matrix[:, pos]
        assert np.allclose(AA_vec_map[self.seq[pos]], v)
        return v

    @lazy_property.LazyProperty
    def matrix(self):
        return np.array([AA_vec_map[aa] for aa in self.seq]).T


class Alignment:
    def __init__(self, seqs: List[str], weight_seqs = True):
        check_seqs(seqs)
        self.seqs = [Sequence(seq) for seq in seqs]
        self.seq_length = len(seqs[0])
        self.n_seqs = len(seqs)
        wts = CalcSeqWeight(seqs) if weight_seqs else np.ones(len(seqs))
        for seq, wt in zip(self.seqs, wts):
            seq.set_wt(wt)
        self.wt = np.sum(wts)
        print('total weight',self.wt)

    def empirical_freqs_single(self, pseudoc):
        """Counts the frequency of each aa at each column in the alignment
        Returns a matrix M[i,a] that gives the number of occurences of
        amino acid a at position i in each sequence.
        :param seqs:
        :param default_freq:
        :param scale:
        :return:
        """
        freqs = np.ones((self.seq_length, 21)) * pseudoc
        # getting empirical frequency of each amino acid in each column
        for i in range(self.seq_length):
            for k in range(self.n_seqs):
                aa = self.seqs[k][i]
                if aa in AA_index_map:
                    aa_index = AA_index_map[aa]
                    freqs[i][aa_index] += self.seqs[k].get_wt()
        return freqs * (1 / ((21 * pseudoc) + self.wt))

    def aas_per_col(self):
        aas_per_col = defaultdict(lambda: defaultdict(set))
        for seq in self.seqs:
            for col, aa in enumerate(seq):
                aa_index = AA_index_map[aa]
                aas_per_col[i].add(aa_index)
        return aas_per_col

    def __len__(self):
        return self.n_seqs

    @property
    def dims(self):
        return self.n_seqs, self.seq_length


def CovarianceMatrix(seqs, pseudoc=0, shrink=False, weight_seqs = True):
    alignment = Alignment(seqs, weight_seqs = weight_seqs)
    pa = alignment.empirical_freqs_single(pseudoc=pseudoc)
    n_seqs, seq_len = alignment.dims
    cov_mat = np.zeros((21 * seq_len, 21 * seq_len))
    for i in range(seq_len):
        for j in range(i, seq_len):
            pab = np.ones((21, 21)) * (pseudoc / 21)
            if i==j:
                pab = np.zeros((21, 21))
                for aa in AA_alphabet:
                    idx = AA_index_map[aa]
                    pab[idx][idx]=pa[i][idx]
            if i != j:
                for k in range(n_seqs):
                    a, b = alignment.seqs[k][i], alignment.seqs[k][j]
                    if a in AA_index_map and b in AA_index_map:
                        a_idx, b_idx = AA_index_map[a], AA_index_map[b]
                        pab[a_idx][b_idx] += alignment.seqs[k].get_wt()
                #renormalize
                pab /= (pseudoc * 21 + alignment.wt)

            for a_ in range(21):
                for b_ in range(21):
                    a, b = AA_index_map[AA_alphabet[a_]], AA_index_map[AA_alphabet[b_]]
                    if i != j or a == b:
                        val = pab[a][b] - pa[i][a] * pa[j][b]
                        cov_mat[i * 21 + a][j * 21 + b] = val
                        cov_mat[j * 21 + b][i * 21 + a] = val
    print('n entries in cov mat :', cov_mat.shape[0] ** 2)
    print('n elts != median', len(cov_mat[cov_mat != np.median(cov_mat)]))
    print('min/max element ', np.min(cov_mat), np.max(cov_mat))
    print('quantiles ', np.quantile(cov_mat, q=np.linspace(0, 1, 10)))

    return cov_mat if not shrink else shrink_cov_mat(cov_mat, wt=alignment.wt)


def seqs_as_mat(seqs):
    m, n = len(seqs[0]), len(seqs)
    seqs = [Sequence(s) for s in seqs]
    print(type(seqs[0]))
    assert seqs[0].matrix.shape == (num_aas, m)
    mat = np.concatenate([s.matrix for s in seqs]).reshape(num_aas * n, m)
    # col sum
    assert np.alltrue(np.sum(mat, axis=0) == n)
    return mat


class AlignmentMatrix:

    def __init__(self, m, default=0):
        # number of AA's per sequence
        self.m = m
        self.num_aas = len(AA_alphabet)
        self.mat = np.ones((num_aas * m, num_aas * m)) * default

    def __setitem__(self, key, value):
        i, j, a, b = key
        a, b = self.as_int(a, b)
        assert 0 <= a < self.num_aas and 0 <= b < self.num_aas
        self.mat[i * num_aas + a, j * num_aas + b] = value

    def __getitem__(self, key):
        i, j, a, b = key
        a, b = self.as_int(a, b)
        return self.mat[i * num_aas + a, j * num_aas + b]

    def as_int(self, a, b):
        if type(a) == str:
            a = AA_index_map[a]
        if type(b) == str:
            b = AA_index_map[b]
        return a, b

    @property
    def matrix(self):
        return self.mat  # lil_matrix(self.mat)


def shrink_cov_mat(cov_mat, wt=1):
    lam_high = 0.99
    lam_low = 0
    mu = np.mean(np.diag(cov_mat))
    print('mu of cov mat diagonal :', mu)
    F = np.zeros(cov_mat.shape)
    np.fill_diagonal(F, mu)
    print('np.mean of F diag', np.mean(np.diag(F)))
    print('initial min eval :', np.min(np.linalg.eigvalsh(cov_mat)))
    while lam_high - lam_low > 1e-2:
        lam = (lam_high + lam_low) / 2
        S_p = lam * F + (1 - lam) * cov_mat
        if not is_pos_defs(S_p):
            lam_low = lam
            # lam_low = lam
        else:
            lam_high = lam

    S_p = lam_high * F + (1 - lam_high) * cov_mat
    print('final', np.min(np.linalg.eigvalsh(S_p)))
    print('lambda', lam_high)
    return S_p


def is_pos_defs(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        raise Exception('matrix not symmetric')


"""
class MSAUtils:

    @lazy_property.LazyProperty
    def AA_vec_map(self):
        AA_vec_map = {}
        for aa, i in AA_index_map.items():
            tmp = np.zeros(num_aas)
            tmp[i] = 1
            AA_vec_map[aa] = tmp
        return AA_vec_map


print(AA_index_map)

seqs = []
n, m = 100, 5
# make n random sequences of length m
for _ in range(n):
    s = ""
    for _ in range(m):
        s += AA_alphabet[np.random.randint(0, len(AA_alphabet))]
    seqs.append(s)
cov_mat = CovarianceMatrix(seqs).matrix
print('first mat')
print(CovarianceMatrix(seqs).matrix)
print(len(cov_mat[cov_mat != 0]))
print(21 * m * 21 * m)
print(cov_mat[0])
print(cov_mat.shape)
print('===')
"""


def average_product_correct(contact_norms):
    s_all = np.mean(contact_norms)
    s_cols = np.mean(contact_norms, axis=0)
    for i in range(len(contact_norms)):
        for j in range(len(contact_norms)):
            contact_norms[i, j] = contact_norms[i, j] - ((s_cols[i] * s_cols[j]) / s_all)


def contact_norms(arr, k=21):
    n = len(arr) // k
    ns = np.zeros((n, n))
    for i in range(n):
        r = i * k
        for j in range(n):
            c = j * k
            for a in range(k):
                for b in range(k):
                    if a != AA_index_map['-'] and b != AA_index_map['-']:
                        ns[i][j] += abs(arr[r + a][c + b])
    return ns


def precision(predicted, native, min_sep=6, max_sep=None, top=None):
    if not top:
        top = [1, 2, 5, 10, 25, 50, 100]
    mat = np.copy(predicted)
    mat[np.tril_indices(len(mat), min_sep)] = 0
    max_sep = max_sep or len(mat)
    mat[np.triu_indices(len(mat),max_sep)]=0
    print('min/max', np.min(mat), np.max(mat))
    if len(mat[mat!=0])==0:
        return None
    native_contacts = np.zeros(native.shape)
    native_contacts[native <= 8] = 1
    #print(len(native_contacts[native_contacts == 0]))
    precision = []
    for t in top:
        predicted_msk = get_contact_mask(mat, t)
        #print(len(predicted_msk[predicted_msk > 0]), t)
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


def get_sparsity(mat, thresh=1e-3):
    temp = np.abs(np.copy(mat))
    np.fill_diagonal(temp,0)
    temp[temp<thresh]=0
    return len(temp[temp>0])/((temp.shape[0]*temp.shape[1])-temp.shape[0])

def scale_spectrum(cov_mats):
    #trace = sum of eigenvalues
    #all matrices are positive definite, so scaling
    #matrices to have same eigenvalue sum may make the
    #inverse cov matrix more similar?
    traces = np.array([np.trace(cov_mat) for cov_mat in cov_mats])
    max_trace = np.argmax(traces)
    scales = max_trace/traces
    return np.array([scale*c for scale,c in zip(scales,cov_mats)])
