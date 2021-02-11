
import os
from utils import parse_seqs
root = '/home/mmcpartlon/suppdata'
max_to_clust = '3'
max_workers = '4'
max_iters = '60'
save_rt = '/mnt/local/mmcpartlon/results_phylo_low_pen_lam01_more_iters'
shrink = '0'
scale = '0'
clust = '0' #0 = no clust, #1 : cd hit, #2: random, #3: most similar to target, #4: phylo
lam,beta = 0.005,0.003
x = '/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata'

s = '-t phylo'
s+=' -d /mnt/local/mmcpartlon/dmats/{fam}_blossum_62.npy'
s+=' -c {cluster_size}'
s+=' -p {max_workers}'
s+=' -i {max_iters}'
s+=' -l {lam}'
s+=' -b {beta}'
s+=' -w {out}'


with open('/mnt/c/Users/mm851/PycharmProjects/InteractionTVGL/args.txt','w+') as fl:
    for fam in os.listdir(x+'/pdb'):
        f = fam.split('.')[0]
        aln_path = f'~/suppdata/converted_aln/{f}.aln.fasta'
        seq_path = f"~/suppdata/seq/{f}.fasta"
        all_seqs = parse_seqs(f'/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata/aln/{f}.aln')
        cluster_size = min(5,max(2,len(all_seqs)//800))
        out = '/mnt/local/mmcpartlon/target_sp_03'
        options = s.format(fam = f,
                           cluster_size = cluster_size,
                           max_workers=max_workers,
                           max_iters=max_iters,
                           lam=lam,
                           beta=beta,
                           out = out)

        fl.write(f'{seq_path} {aln_path} {options} >{f}_out.log &\n')



