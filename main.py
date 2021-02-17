import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description="Interaction TVGL",
                        epilog='output format is a python dict, of the form'
                               ' {theta_set : List[np.ndarray],'
                               ' clusters : List[List[sequences]],cov_mats'
                               ' : List[np.ndarray], seq: sequence,'
                               ' args : dict)',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('target_seq',
                    help="The path to the target sequence (.fasta)",
                    type=str)
parser.add_argument('aln',
                    help="path to MSA alignment file",
                    type=str)
parser.add_argument('-f', '--target_sparsity',
                    help="target sparsity for precision matrix (as decimal)",
                    type=float,
                    default=0.03)
parser.add_argument('-u', '--prec_thresh',
                    help="threshold to use when processing precision matrix."
                         "values P[i,j] with abs(P[i,j])<thresh will be set to 0",
                    type=float,
                    default=1e-3)
parser.add_argument('-t', '--cluster_type',
                    help="Method to use for clustering sequences in MSA",
                    choices=['phylo', 'cdhit'],
                    default='phylo')
parser.add_argument('-e', '--phylo_path',
                    help='Path to phylogenetic tree file for MSA. If the file in'
                         ' the given location does not exist, a phylogenetic tree '
                         'will be produced, and stored in this location under the '
                         'given name',
                    type=str)
parser.add_argument('-d', '--dist_path',
                    help='Path to alignment pairwise distance matrix (used in the construction of '
                         'phylogenetic tree). If the file at the given location does not exist,'
                         'the pairwise distances will be saved at this location.',
                    type=str)
parser.add_argument('-g', '--cluster_path',
                    help='path to a cluster file to use for generating sample'
                         ' covariance matrices.'
                         ' The file format must be .npy, and should contain a'
                         ' list of lists of sequences,'
                         ' representing the clusters. Sequences must match with'
                         ' those of the MSA file'
                         ' (i.e. gaps should not be removed). All cluster related'
                         ' options will be ignored'
                         ' is specified',
                    type=str)
parser.add_argument('-a', '--all_seq_clust',
                    help="Set the first cluster sample covariance matrix to that"
                         " of the entire MSA (defaults to"
                         " true). The total number of clusters will still be"
                         " -n_clusters",
                    type=int,
                    default=1,
                    choices=[0, 1])
parser.add_argument('-w', '--output_dir',
                    help="Directory in which output should"
                         " be saved", type=str, default='./')
parser.add_argument('-c', '--n_clusters',
                    help="The number of clusters to use when generating"
                         " precision matrices",
                    default=4,
                    type=int)
parser.add_argument('-p', '--max_processors',
                    help="Maximum number of processors to allocate"
                         " (-1 for all available)",
                    type=int, default=-1)
parser.add_argument('-y', '--max_threads',
                    help="Maximum number of threads to allocate"
                         " per-processor (-1 for all available)",
                    type=str, default='4')
parser.add_argument('-i', '--max_iters',
                    help="Maximum number of iterations to run ADMM updates",
                    type=int,
                    default=150)
parser.add_argument('-s', '--shrink',
                    help="Condition sample covariance matrices C_shrink = (1-lam)*"
                         "I+lam*C where lam is chosen "
                         "to enforce PD of C_shrink (default = False)",
                    action="store_true",
                    default=False)
parser.add_argument('-l', '--lamb',
                    help='Value(s) for lambda parameter to TVGL (controls'
                         ' sparsity of precision matrix - larger =>'
                         ' more sparse). If one value is given, then the same'
                         ' lambda value is used for each sample'
                         ' covaraince matrix. Otherwise, a lambda value must '
                         'be given for each sample covariance matrix',
                    default=[0.005],
                    nargs='+',
                    type=float)
parser.add_argument('-b', '--beta',
                    help='Value(s) for beta parameter to TVGL (controls consistency'
                         ' between output precision'
                         ' matrices - high beta => more similarity). If one value is'
                         ' given, then the same beta'
                         ' value is used for each sample covaraince matrix. Otherwise,'
                         ' a beta value must be given'
                         ' for consecutive pair of sample covariance matrices',
                    default=[0.0025],
                    nargs='+',
                    type=float)
parser.add_argument('-q', '--save_intermediate',
                    help='whether or not to save intermediate results '
                         '(while searching for optimal parameters)',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-r', '--save_cov_mats',
                    help='Whether the sample covaraince matrices should be saved in'
                         ' the output (defaults to true)',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-v', '--verbose',
                    help='level of verbosity : prints status/ current iteration of'
                         ' algorithm, time taken to complete admm steps',
                    default=1,
                    choices=[0, 1])
parser.add_argument('-o', '--cdhit_options',
                    help='options to pass to cdhit '
                         'clustering program', nargs='+', type=str,
                    default='-c 0.8, -G 1 -n 5 -g 0 -aS 0.85 -aL 0'.split(' '))
args = parser.parse_args()
_n_threads = args.max_threads
if int(_n_threads) > 0:
    # os.environ["OMP_NUM_THREADS"] = _n_threads  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = _n_threads  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = _n_threads  # export MKL_NUM_THREADS=6
    # os.environ["VECLIB_MAXIMUM_THREADS"] = _n_threads  # export VECLIB_MAXIMUM_THREADS=4
    # os.environ["NUMEXPR_NUM_THREADS"] = _n_threads  # export NUMEXPR_NUM_THREADS=6

# have to import after setting number of threads
from cluster.cd_hit_cluster import *
from cluster.phylo_cluster import gen_phylo_clusters
import TVGL_Interaction
from TVGL_Interaction import TVGL, PenaltyType
import sys
import time
from utils.utils import *
import importlib
import multiprocessing
from utils.bio_utils import CovarianceMatrix

target_sparsity = args.target_sparsity
penalty = PenaltyType.L1

# Set the TVGL ADMM parameters
lamb = args.lamb
beta = args.beta

if len(args.lamb) > 1:
    if len(args.lamb) != args.n_clusters:
        print('lambda values must be specified for'
              ' each sample covariance matrix')
        print(f'got --lamb : {args.lamb}, and '
              f'--n_clusters : {args.n_clusters}')
        sys.exit(1)
else:
    lamb = args.lamb * args.n_clusters

if len(args.beta) > 1:
    if len(args.lamb) != args.n_clusters - 1:
        print('beta values must be specified for each pair'
              ' of sample covariance matrix')
        print(f'got --lamb : {args.lamb}, and --n_clusters '
              f': {args.n_clusters}, => n_pairs'
              f' : {args.n_clusters - 1}')
        sys.exit(1)
else:
    beta = args.beta * (args.n_clusters - 1)

ptn_name = args.target_seq.split('/')[-1].split('.')[0]

# Set up output directories
os.makedirs(args.output_dir, exist_ok=True)
if args.phylo_path:
    os.makedirs(os.path.dirname(args.phylo_path), exist_ok=True)

target_seq = parse_sequence(args.target_seq)

use_phylo = args.cluster_type == 'phylo'
display_message(f"Beginning clustering of protein"
                f" sequences using method {args.cluster_type}",
                1, args.verbose)
if args.cluster_path:
    clusters = {i: clust for i, clust in enumerate(load_npy(args.cluster_path))}
    args.n_clusters = len(clusters)
    args.all_seq_clust = False
else:
    if not use_phylo:
        clusters = gen_cdhit_clusters(args.aln, output_cluster_path='./tmp',
                                      cd_hit_options=' '.join(args.cdhit_options))
    else:
        n_to_clust = args.n_clusters - 1 if args.all_seq_clust else args.n_clusters
        clusters = gen_phylo_clusters(args.aln,
                                      n_to_clust,
                                      dist_path=args.dist_path,
                                      out=args.phylo_path,
                                      verbose=args.verbose)
display_message("finished clustering of protein sequences...",
                1, args.verbose)

n_seqs = sum(len(c) for c in clusters.values())
display_message(f"generated {len(clusters)} clusters"
                f" from {n_seqs} sequences", 1, args.verbose)

n_cpus = multiprocessing.cpu_count()
n_processors = n_cpus if args.max_processors < 0 else args.max_processors
n_processors = min(n_processors, n_cpus, args.n_clusters + 1)

all_seqs = [seq for cluster in clusters.values() for seq in cluster]

if args.all_seq_clust:
    temp = {0: all_seqs}
    for i, clust in enumerate(clusters.values()):
        if len(temp) == args.n_clusters:
            break
        temp[i + 1] = clust
    clusters = temp

display_message(f"Beginning generation of covariance"
                f" matrices for clusters 1..{args.n_clusters}",
                1, args.verbose)
print('final cluster sizes :', [len(c) for c in clusters.values()])
cov_mats = [CovarianceMatrix(c, pseudoc=1, shrink=args.shrink)
            for c in clusters.values()]
display_message(f"finished generation of covariance"
                f" matrices for clusters", 1, args.verbose)

prev_lambs, prev_betas = [], []
stop = False
MAX_TRIES = 8
use_cluster = False
prev_sparsities = []
rfacts = [0] * len(clusters)

print(f'using lambda : {lamb}, and beta {beta}')
for i in range(MAX_TRIES):
    importlib.reload(TVGL_Interaction)
    importlib.reload(multiprocessing)
    importlib.reload(np)
    start = time.time()
    sp = os.path.join(args.output_dir, f"{ptn_name}_{i}.npy")
    assert os.path.exists(os.path.dirname(sp))
    params_converged = False
    print('lam,', lamb, 'beta', beta)
    if len(prev_sparsities) > 0:
        sparsity_diff = np.abs(np.array(prev_sparsities[-1]) - target_sparsity)
        params_converged = np.alltrue(sparsity_diff < 0.3 * 1e-2)
        # lam_diff = np.array(lamb)-np.array(prev_lambs[-1])
        # params_converged = params_converged or np.alltrue(np.abs(lam_diff)<1e-5)
    if params_converged or i == MAX_TRIES - 1:
        theta_set = TVGL(np.copy(cov_mats),
                         n_processors=n_processors,
                         lamb=lamb,
                         beta=beta,
                         indexOfPenalty=penalty,
                         max_iters=150,
                         verbose=True,
                         use_cluster=use_cluster)
        stop = True
        sp = os.path.join(args.output_dir, f"{ptn_name}_final.npy")
    else:
        theta_set = TVGL(np.copy(cov_mats),
                         n_processors=n_processors,
                         lamb=lamb,
                         beta=beta,
                         max_iters=args.max_iters,
                         indexOfPenalty=penalty,
                         verbose=True,
                         use_cluster=use_cluster)
    display_message(f'finished generating precision matrices.'
                    f' time : {(time.time() - start)} s',
                    1,
                    args.verbose)
    data = {'theta set': theta_set,
            'clusters': clusters,
            'lam': lamb,
            'beta': beta,
            'seq': target_seq,
            'args': args.__dict__}
    if args.save_cov_mats:
        data['cov_mats'] = cov_mats
    if args.save_intermediate or stop:
        np.save(sp, data)
    display_message(f"finished tvgl", 1, args.verbose)

    l, b, s = adjust_lam_and_beta(theta_set,
                                  lamb,
                                  beta,
                                  prev_sparsities,
                                  prev_lambs,
                                  prev_betas,
                                  target_sparsity=target_sparsity,
                                  thresh=args.prec_thresh)

    prev_lambs.append(list(lamb))
    prev_betas.append(list(beta))
    prev_sparsities.append(list(s))
    lamb, beta = list(l), list(b)
    display_message(f"previous lambs : {prev_lambs}")
    display_message(f"previous betas : {prev_betas}")
    display_message(f"sparsities : {prev_sparsities}")
    display_message(f"current lamb : {lamb}")
    display_message(f"current beta : {beta}")

    if stop:
        break
