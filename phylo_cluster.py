from collections import defaultdict
#import treeCl
import numpy as np
import os
import sys
from functools import partial
from typing import Callable
import time
from utils import display_message,load_npy, parse_seqs

"""
phylo_path = './test_phylo.dnd'
if not os.path.exists(phylo_path):
    input_dir = '/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata/phy/'
    c = treeCl.Collection(input_dir=input_dir, file_format='fasta')
    c.calc_trees(executable='raxmlHPC-AVX',  # specify raxml binary to use
                 threads=8,  # use multithreaded raxml
                 model='PROTGAMMAWAGX',  # this model of evolution
                 fast_tree=True)  # use raxml's experimental fast tree search option
    print('finished tree calc!')

    with open('test_phylo.dnd', 'w+') as f:
        f.write(c.trees[0])

from ete3 import Tree


t = Tree(phylo_path)
root = t.get_tree_root()
print(root)
print(root.children)
print(root.get_children())
print(t.get_distance(root, root.children[1]))
print(t.get_leaves())

"""

# Remove rows and columns of matrix with the listed indices
withoutIndices = lambda m, ids: np.delete(np.delete(m, ids, axis=0), ids, axis=1)

def get_inter_group_dists(tree, a_nodes, b_nodes, dmap=None):
    if dmap is None:
        return [tree.get_distance(a, b) for a in a_nodes for b in b_nodes]
    else:
        return [dmap[a][b] for a in a_nodes for b in b_nodes]

def max_linkage(a_nodes, b_nodes, tree, dmap=None):
    return np.max(get_inter_group_dists(tree, a_nodes, b_nodes, dmap=dmap))

def quantile_linkage(a_nodes, b_nodes, tree, q=0.8, dmap=None):
    return np.quantile(get_inter_group_dists(tree, a_nodes, b_nodes, dmap=dmap), q=q)

def avg_linkage(a_nodes, b_nodes, tree, dmap=None):
    return np.mean(get_inter_group_dists(tree, a_nodes, b_nodes, dmap=dmap))


def get_node_dist_map(tree):
    dmap = defaultdict(lambda: defaultdict(int))
    nodes = [x for x in tree.get_leaves()]
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            dmap[a][b] = dmap[b][a] = tree.get_distance(a, b)
    return dmap

def partition_tree(tree, n_clusters, linkage_fn: Callable, dmap = None):
    clusters = [[x] for x in tree.get_leaves()]
    if dmap is None:
        dmap = get_node_dist_map(tree)
    return partition_helper(clusters, n_clusters, partial(linkage_fn, tree=tree, dmap = dmap))

def partition_dists(dists, n_clusters, linkage_fn: Callable):
    clusters = [[x] for x in range(len(dists))]
    dists = np.array(dists)
    np.fill_diagonal(dists, 1e10)
    return partition_helper(clusters, n_clusters, partial(linkage_fn, tree=None, dmap=dists), dists)

def _partition_dists(dists, n_clusters, clusters = None):
    if not clusters:
        clusters = [[x] for x in range(len(dists))]
    dists = np.array(dists)
    np.fill_diagonal(dists,1e10)
    return partition_helper(clusters, n_clusters, dists, ret_dists = True)

def partition_scores(dist_scores, clusters):
    #calculate the sizes in the resulting partition if clusters i and j are merged
    n = len(dist_scores)
    N = sum([len(c) for c in clusters])
    scale = n*(np.log(N)-np.log(n))
    S = sum([np.log(len(c)) for c in clusters])
    deltas = np.zeros(dist_scores.shape)
    for i in range(n):
        xi = len(clusters[i])
        for j in range(i+1,n):
            xj = len(clusters[j])
            log_change = S - np.log(xi)-np.log(xj) + np.log(xi+xj)
            deltas[i][j] = np.exp(log_change-scale)
    deltas+=np.transpose(deltas)
    deltas/=np.mean(deltas)
    deltas/=np.max(deltas)
    return dist_scores - deltas


def partition_helper(clusters, n_clusters, d, ret_dists = False):
    n = len(clusters)
    assert d.shape[0]==d.shape[1]==n
    tmp = d
    if n<100:
        tmp = partition_scores(np.copy(d), clusters)
    i, j = np.unravel_index(tmp.argmin(), tmp.shape)
    d[i,:]=d[:,i]=np.maximum(d[:,i],d[:,j])
    d[i,i]=1e10
    d = withoutIndices(d,j)
    clusters[i].extend(clusters[j])
    del clusters[j]
    if len(clusters)==n_clusters:
        if ret_dists:
            return d, clusters
        else:
            return clusters
    return partition_helper(clusters, n_clusters, d, ret_dists = ret_dists)

#from cogent3 import load_aligned_seqs
#from cogent3.evolve import distance
#from cogent3.evolve.models import JTT92


from bio_utils import AA_index_map
def rewrite_seqs(aln_path, save_path):
    seqs = []
    headers = []
    with open(aln_path, 'r+') as f:
        for x in f:
            if not x.startswith('>') and len(x) > 1:
                seq = x.strip()
                tmp = ''
                for s in seq:
                    if s not in AA_index_map:
                        tmp+='-'
                    else:
                        tmp+=s
                seqs.append(tmp+'\n')
            elif len(x)>1:
                headers.append(x)
    with open(save_path,'w+') as f:
        for h,s in zip(headers,seqs):
            f.write(h)
            f.write(s)


from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO

def gen_phylo_clusters(aln_path, k = None, max_k = 4, out = None, dist_path = None, verbose = 1):
    seqs = parse_seqs(aln_path)
    k = k or min(max_k,max(2,len(seqs)//600))
    #sequences have to be rewritten in a specific format to work with alignment reader
    nm = str(np.random.randint(1e10))
    rewrite_path = f"./{nm}.fasta"
    if not os.path.exists(rewrite_path):
        rewrite_seqs(aln_path,rewrite_path)

    aln = AlignIO.read(open(rewrite_path), 'fasta')
    display_message('starting distance computation',1,verbose=verbose)
    start = time.time()
    nm = str(np.random.randint(1e10))
    ds_path = dist_path or f"./{nm}.fasta"
    if not os.path.exists(ds_path):
        os.makedirs(ds_path,exist_ok=True)
        calculator = DistanceCalculator('blosum62')
        ds = calculator.get_distance(aln)
        np.save(ds_path, ds)
    #remove the (temporary) alignment file
    os.remove(rewrite_path)
    #remove the distance file if no path was specified
    if not dist_path:
        os.remove(ds_path)
    display_message(f'finished distance computation {(time.time()-start)/60} mins',1,verbose)
    display_message('starting sequence clustering',1,verbose)
    start = time.time()
    ds = load_npy(ds_path)
    clusters = [[s] for s in seqs]
    interval = 50
    for _k in range(1,(len(seqs)//interval)-1):
        ds, clusters = _partition_dists(ds, max(10,len(seqs)-_k*interval), clusters=clusters)
        progress = np.round(100*(_k/((len(seqs)//interval)-2)),1)
        display_message(f'clustering... progress : {progress} %',1,verbose)
    ds, clusters = _partition_dists(ds, k, clusters)
    display_message(f'finished clustering sequences {(time.time()-start)/60} mins',1,verbose)
    display_message(f'cluster sizes : {[len(c) for c in clusters]}', 1, verbose)
    return {i:c for i,c in enumerate(clusters)}


