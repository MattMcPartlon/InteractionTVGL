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
                    choices=['phylo', 'cdhit', 'random'],
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
parser.add_argument('-j', '--max_trial_runs',
                    help="maximum number of runs to adjust parameters - parameters"
                         "are adjusted after each trial run so that the sparsity of"
                         "the returned precision matrices match with --target_sparsity",
                    type=int,
                    default=10)
parser.add_argument('-k', '--log_file',
                    help="file to write logging information to (defaults to results directory)",
                    type=str,
                    default='')
parser.add_argument('-v', '--verbose',
                    help='level of verbosity : prints status/ current iteration of'
                         ' algorithm, time taken to complete admm steps',
                    default=1,
                    choices=[0, 1])
parser.add_argument('-o', '--cdhit_options',
                    help='options to pass to cdhit '
                         'clustering program', nargs='+', type=str,
                    default='-c 0.8, -G 1 -n 5 -g 0 -aS 0.85 -aL 0'.split(' '))
parser.add_argument('-m', '--max_gapf',
                    help='maximum allowed gap frequency for column in MSA',
                    type=float,
                    default=1)
parser.add_argument('-z', '--beta_ratio',
                    help='(optional) choose beta as lam[1]*beta_ratio at each step. Only applied'
                         'if >0',
                    type=float,
                    default=0)
parser.add_argument('-x', '--penalty_ty',
                    help='penalty type to use for TVGL procedure (L1 = 1,\n L2=2,\n LAPLACIAN=3,\n PERTURBATION=5)',
                    type=int,
                    default=1)
parser.add_argument('-n', '--remove_gap_posns',
                    help='how to handle positions exceeding gap threshold',
                    type=int,
                    default=0)
parser.add_argument('-s', '--single_lam',
                    help='use only one lambda value, shared for all clusters. Lambda is set so that'
                         'the first matrix has target sparsity --target_sparsity, and others are ignored.',
                    type=int,
                    default=0)
args = parser.parse_args()
_n_threads = args.max_threads
if int(_n_threads) > 0:
    # os.environ["OMP_NUM_THREADS"] = _n_threads  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = _n_threads  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = _n_threads  # export MKL_NUM_THREADS=6
    # os.environ["VECLIB_MAXIMUM_THREADS"] = _n_threads  # export VECLIB_MAXIMUM_THREADS=4
    # os.environ["NUMEXPR_NUM_THREADS"] = _n_threads  # export NUMEXPR_NUM_THREADS=6

# have to import after setting number of threads
import logging
import sys
ptn_name = args.target_seq.split('/')[-1].split('.')[0]
log_file = args.log_file or os.path.join(args.output_dir, f'{ptn_name}_error_log.log')
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except:
        pass
if args.log_file != '':
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
else:
    logging.basicConfig(stream = sys.stdout, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

import traceback
from cluster.cd_hit_cluster import *
from cluster.phylo_cluster import gen_phylo_clusters
import TVGL_Interaction
from TVGL_Interaction import TVGL, PenaltyType
import time
from utils.utils import *
import importlib
import multiprocessing
from utils.bio_utils import CovarianceMatrix, Alignment
from cluster.random_cluster import random_partition



target_sparsity = args.target_sparsity
penalty = PenaltyType.L1
if args.penalty_ty == 2:
    penalty = PenaltyType.L2
if args.penalty_ty == 3:
    penalty = PenaltyType.LAPLACIAN
if args.penalty_ty == 4:
    penalty = PenaltyType.LINF
if args.penalty_ty == 5:
    penalty = PenaltyType.PERTURBATION

# Set the TVGL ADMM parameters
lamb = args.lamb
beta = args.beta

if len(args.lamb) > 1:
    if len(args.lamb) != args.n_clusters:
        logger.error('lambda values must be specified for'
                     ' each sample covariance matrix')
        logger.error(f'got --lamb : {args.lamb}, and '
                     f'--n_clusters : {args.n_clusters}')
        sys.exit(1)
else:
    lamb = args.lamb * args.n_clusters

if len(args.beta) > 1:
    if len(args.lamb) != args.n_clusters - 1:
        logger.error('beta values must be specified for each pair'
                     ' of sample covariance matrix')
        logger.error(f'got --lamb : {args.lamb}, and --n_clusters '
                     f': {args.n_clusters}, => n_pairs'
                     f' : {args.n_clusters - 1}')
        sys.exit(1)
else:
    beta = args.beta * (args.n_clusters - 1)

if args.beta_ratio > 0:
    beta = [lamb[0] * args.beta_ratio] * len(beta)





# Set up output directories
os.makedirs(args.output_dir, exist_ok=True)
if args.phylo_path:
    os.makedirs(os.path.dirname(args.phylo_path), exist_ok=True)

target_seq = parse_sequence(args.target_seq)

logger.info(f"Beginning clustering of protein"
            f" sequences using method {args.cluster_type}")
if args.cluster_path:
    clusters = {i: clust for i, clust in enumerate(load_npy(args.cluster_path))}
    args.n_clusters = len(clusters)
    args.all_seq_clust = False

else:
    n_to_clust = args.n_clusters - 1 if args.all_seq_clust else args.n_clusters
    clusters = {}
    if n_to_clust >0:
        if args.cluster_type == 'cdhit':
            clusters = gen_cdhit_clusters(args.aln, output_cluster_path='./tmp',
                                          cd_hit_options=' '.join(args.cdhit_options))
        # clusters are sorted by max distance to target sequence
        elif args.cluster_type == 'phylo':
            clusters = gen_phylo_clusters(args.aln,
                                          target_seq,
                                          k=n_to_clust,
                                          dist_path=args.dist_path,
                                          )
        else:  # random
            clusters = random_partition(args.aln, n_to_clust)

logger.info("finished clustering of protein sequences...")

n_seqs = sum(len(c) for c in clusters.values())
logger.info(f"generated {len(clusters)} clusters"
            f" from {n_seqs} sequences")

n_cpus = multiprocessing.cpu_count()
n_processors = n_cpus if args.max_processors < 0 else args.max_processors
n_processors = min(n_processors, n_cpus, args.n_clusters + 1)

all_seqs = parse_seqs(args.aln)

if args.all_seq_clust:
    temp = {0: all_seqs}
    for i, clust in enumerate(clusters.values()):
        if len(temp) == args.n_clusters:
            break
        temp[i + 1] = clust
    clusters = temp


shrink = False
logger.info(f'n_clusters, {len(clusters)}')
logger.info(f'len(clusters[0] {len(clusters[0])}')
logger.info(f"Beginning generation of covariance"
            f" matrices for clusters 1..{args.n_clusters}")
szs = [len(c) for c in clusters.values()]
logger.info(f'final cluster sizes : {szs}', )
cov_mats = [CovarianceMatrix(c, pseudoc=1, shrink=shrink)
            for c in clusters.values()]
logger.info(f"finished generation of covariance matrices for clusters")

prev_lambs, prev_betas = [], []
stop = False
MAX_TRIES = args.max_trial_runs
prev_sparsities = []
target_sparsity += 0.001  # add tol bc final run produces more sparse output
best_lam, best_beta, best_diff = lamb, beta, 1
idxs = np.arange(len(clusters)).astype(int)
if args.single_lam == 1:
    idxs = np.array([0])
#sc = 0.7
#ws = np.array([sc**i for i in range(len(lamb))])
withoutIndices = lambda m, ids: np.delete(np.delete(m, ids, axis=0), ids, axis=1)
alns = [Alignment(c) for c in clusters.values()]
gap_fs = [aln.gap_freqs() for aln in alns]
max_gapf = args.max_gapf
max_gap_posns = set()
og_size = len(cov_mats[0])
remove_posns = args.remove_gap_posns == 1
for i, fs in enumerate(gap_fs):
    violations = np.arange(len(fs))[fs > max_gapf]
    for v in violations:
        max_gap_posns.add(int(v))
remove_idxs = []
for pos in max_gap_posns:
    temp = pos*21+np.arange(21).astype(int)
    remove_idxs.extend(list(temp))
if remove_posns:
    #remove positions all together
    for i in range(len(cov_mats)):
        cov_mats[i]=withoutIndices(cov_mats[i], remove_idxs)

else:
    for i,fs in enumerate(gap_fs):
        violations = np.arange(len(fs))[fs>max_gapf]
        for v in violations:
            s = 21*v
            diag = cov_mats[i][s:s + 21, s:s+21]
            cov_mats[i][s:s+21,:] = 0
            cov_mats[i][:,s:s + 21] = 0
            cov_mats[i][s:s + 21, s:s + 21] = diag

#remove these columns from covariance matrices?
#mask these columns?

logger.info(f'gap posns removed : {remove_idxs}')
logger.info(f'num gap posns removed : {len(remove_idxs)/21}')

ws = [aln.wt for aln in alns]
ws /= np.sum(ws)
ws/=np.min(ws)
tol = 1.5*1e-3

logger.info(f'using ws : {ws}')
try:
    for i in range(MAX_TRIES):
        importlib.reload(TVGL_Interaction)
        importlib.reload(multiprocessing)
        importlib.reload(np)
        start = time.time()
        sp = os.path.join(args.output_dir, f"{ptn_name}_{i}.npy")
        assert os.path.exists(os.path.dirname(sp))
        params_converged = False
        if len(prev_sparsities) > 0:
            prs = np.array(np.array(prev_sparsities[-1]))
            sparsity_diff = np.array(np.abs(prs - target_sparsity)[idxs])
            params_converged = np.alltrue(sparsity_diff < tol)
            # lam_diff = np.array(lamb)-np.array(prev_lambs[-1])
            # params_converged = params_converged or np.alltrue(np.abs(lam_diff)<1e-5)
            if np.max(sparsity_diff)<best_diff:
                best_diff = np.max(sparsity_diff)
                best_lam = list(prev_lambs[-1])
                best_beta = list(prev_betas[-1])
        if params_converged or i == MAX_TRIES - 1:
            theta_set = TVGL(np.copy(cov_mats),
                             n_processors=n_processors,
                             lamb=best_lam,
                             beta=best_beta,
                             indexOfPenalty=penalty,
                             w=ws,
                             max_iters=120,
                             verbose=True)
            stop = True
            sp = os.path.join(args.output_dir, f"{ptn_name}_final.npy")
        else:
            theta_set = TVGL(np.copy(cov_mats),
                             n_processors=n_processors,
                             lamb=lamb,
                             beta=beta,
                             indexOfPenalty=penalty,
                             w=ws,
                             max_iters=args.max_iters,
                             verbose=True)
        logger.info(f'finished generating precision matrices.'
                    f' time : {(time.time() - start)} s')
        l, b, s = adjust_lam_and_beta(theta_set,
                                      lamb,
                                      beta,
                                      prev_sparsities,
                                      prev_lambs,
                                      prev_betas,
                                      target_sparsity=target_sparsity,
                                      thresh=args.prec_thresh,
                                      single_lam = args.single_lam == 1,
                                      idx = 0,
                                      )

        prev_lambs.append(list(lamb))
        prev_betas.append(list(beta))
        prev_sparsities.append(list(s))
        lamb, beta = list(l), list(b)
        if args.beta_ratio>0:
            beta = [lamb[0]*args.beta_ratio]*len(beta)
        logger.info(f"previous lambs : {prev_lambs}")
        logger.info(f"previous betas : {prev_betas}")
        logger.info(f"sparsities : {prev_sparsities}")
        logger.info(f"current lamb : {lamb}")
        logger.info(f"current beta : {beta}")

        if remove_posns:
            for i in range(len(theta_set)):
                theta_set[i] = recover_mat(theta_set[i], og_size, set(remove_idxs))

        data = {'theta set': theta_set,
                'clusters': clusters,
                'lam': prev_lambs[-1],
                'beta': prev_betas[-1],
                'seq': target_seq,
                'args': args.__dict__,
                'sparsity':prev_sparsities[-1]}
        if args.save_cov_mats:
            data['cov_mats'] = cov_mats
        if args.save_intermediate or stop:
            np.save(sp, data)
        logger.info(f"finished tvgl")

        if stop:
            break
except Exception as e:
    print('caught exception :', e)
    tb = traceback.format_exc()
    logger.error(e)
    logger.error(tb)
    #flush the loggers buffer before exiting
    logger.handlers[0].flush()
    raise e
