
import os
from utils.utils import parse_seqs
root = '/home/mmcpartlon/suppdata'
max_to_clust = '3'
max_workers = '4'
max_iters = '60'
save_rt = '/mnt/local/mmcpartlon/results_target_sp_03_larger_clust_size_beta_small_gp_inv'
shrink = 0
scale = 0
clust = 0 #0 = no clust, #1 : cd hit, #2: random, #3: most similar to target, #4: phylo
lam = 0.002
save_int = 0
beta = .002
sp = 0.03
x = '/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata'

s = '-t phylo'
s+=' -d /mnt/local/mmcpartlon/dmats/{fam}_blossum_62.npy'
s+=' -c {cluster_size}'
s+=' -p {max_workers}'
s+=' -i {max_iters}'
s+=' -l {lam}'
s+=' -b {beta}'
s+=' -w {out}'
s+=' -q {save_int}'
s+=' -f {sp}'


with open('/mnt/c/Users/mm851/PycharmProjects/InteractionTVGL/args.txt','w+') as fl:
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
                           out = save_rt+'/results')

        fl.write(f'{seq_path} {aln_path} {options} >{save_rt}/logs/{f}_out.log &\n')



