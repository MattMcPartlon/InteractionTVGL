from random import shuffle

from utils.utils import parse_seqs


def random_partition(msa_path, n_clusters):
    seqs = parse_seqs(msa_path)
    shuffle(seqs)
    n = len(seqs)
    chunk_size = (n // n_clusters) + 1
    return {i : seqs[i * chunk_size:min((i + 1) * chunk_size, n)] for i in range(n_clusters)}
