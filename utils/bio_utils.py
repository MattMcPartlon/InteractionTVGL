from collections import defaultdict
from typing import List

import lazy_property
import numpy as np

from utils.msa_utils import CalcSeqWeight
import logging
logger = logging.getLogger(__name__)

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
    logger.info(f'n entries in cov mat :, {cov_mat.shape[0] ** 2}')
    logger.info(f'min/max element , {np.min(cov_mat)}, {np.max(cov_mat)}')
    logger.info(f'quantiles , {np.quantile(cov_mat, q=np.linspace(0, 1, 10))}')

    return cov_mat if not shrink else shrink_cov_mat(cov_mat, wt=alignment.wt)


def shrink_cov_mat(cov_mat, wt=1):
    lam_high = 1-1e-3
    lam_low = 0
    mu = np.mean(np.diag(cov_mat))
    F = np.zeros(cov_mat.shape)
    np.fill_diagonal(F, mu)
    while lam_high - lam_low > 1e-2:
        lam = (lam_high + lam_low) / 2
        S_p = lam * F + (1 - lam) * cov_mat
        if not is_pos_defs(S_p):
            lam_low = lam
        else:
            lam_high = lam

    S_p = lam_high * F + (1 - lam_high) * cov_mat
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