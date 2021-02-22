
import os
from utils.utils import parse_seqs
root = '/home/mmcpartlon/suppdata'
max_to_clust = '3'
max_workers = '4'
max_iters = '30'
save_rt = '/mnt/local/mmcpartlon/results_target_sp_032_beta_0005_fix_phylo_gpr_inv_first_only'
shrink = 0
scale = 0
clust = 0 #0 = no clust, #1 : cd hit, #2: random, #3: most similar to target, #4: phylo
lam = 0.002
save_int = 1
beta = .0005
sp = 0.032
zero_thresh = 1e-2
clust_ty = 'phylo'
max_trial_runs = 10
x = '/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata'
s = '-t {clust_ty}'
s+=' -d /mnt/local/mmcpartlon/dmats/{fam}_blossum_62.npy'
s+=' -c {cluster_size}'
s+=' -p {max_workers}'
s+=' -i {max_iters}'
s+=' -l {lam}'
s+=' -b {beta}'
s+=' -w {out}'
s+=' -q {save_int}'
s+=' -f {sp}'
s+=' -j {max_trial_runs}'
s+=' -k {save_rt}/logs/{fam}_logger.log'

print(os.getcwd())
with open('./args.txt','w+') as fl:
    print(fl)
    for fam in os.listdir(x+'/pdb'):
        f = fam.split('.')[0]
        aln_path = f'~/suppdata/converted_aln/{f}.aln.fasta'
        seq_path = f"~/suppdata/seq/{f}.fasta"
        all_seqs = parse_seqs(f'/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata/aln/{f}.aln')
        cluster_size = min(5,max(3,len(all_seqs)//1100))
        max_workers = cluster_size
        options = s.format(fam = f,
                           cluster_size = cluster_size,
                           max_workers=max_workers,
                           max_iters=max_iters,
                           lam=lam,
                           beta=beta,
                           sp = sp,
                           save_int = save_int,
                           out = save_rt+'/results',
                           clust_ty = clust_ty,
                           max_trial_runs = max_trial_runs,
                           save_rt = save_rt)

        fl.write(f'{seq_path} {aln_path} {options} >{save_rt}/logs/{f}_out.log &\n')



