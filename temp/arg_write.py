import sys
arg_f_name = 'args.txt' if len(sys.argv)==1 else sys.argv[1]
import os
from utils.utils import parse_seqs
root = '/home/mmcpartlon/suppdata'
max_to_clust = '3'
max_workers = '4'
max_iters = '30'
save_rt = '/mnt/local/mmcpartlon/results_target_sp_03_beta_ratio_05_beta_scale_1_phylo_gpr_inv_weighted_by_clust_3_clust_LINF_lam_1'
shrink = 0
scale = 0
clust = 0 #0 = no clust, #1 : cd hit, #2: random, #3: most similar to target, #4: phylo
lam = 0.004
save_int = 0
#LINF 0.004
#L1 0.0005
beta = 0.0005
sp = 0.03
zero_thresh = 0.001
clust_ty = 'phylo'
max_trial_runs = 12
beta_ratio = 0.3
beta_scale = 1
penalty_ty = 4
all_seq_pos = 0
one_lam = 1
min_clust, max_clust = 3,5
clust_denom = 1100
def fmt_float(f):
    return str(f).replace(".", "_")

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
s+=' -z {beta_ratio}'
s+=' -m {beta_scale}'
s+= ' -x {penalty_ty}'
s+= ' -n {all_seq_pos}'
s+= ' -s {one_lam}'
s+= ' -u {zero_thresh}'

save_rt = f'/mnt/local/mmcpartlon/results_sp_{fmt_float(sp)}_beta_ratio_{fmt_float(beta_ratio)}_beta_scale_{beta_scale}'
save_rt+=f'_{clust_ty}_gpr_inv_weighted_by_clust_min_{min_clust}_max_{max_clust}_penalty_ty_{penalty_ty}_lam_const' \
         f'_{one_lam}_thresh_{fmt_float(zero_thresh)}_all_seq_pos_{all_seq_pos}_clust_denom_{clust_denom}_DYNAMIC_gap_removed_final'

target_list = '1a3aA	1a6mA	1aapA	1abaA	1atzA	1bebA	1behA	1brfA	1c44A	1c9oA	1cc8A	1chdA	1ctfA	1cxyA	1d1qA	1dbxA'
targets = target_list.split('\t')
save_path = f'/mnt/c/Users/mm851/PycharmProjects/InteractionTVGL/temp/{arg_f_name}'

def get_const_lam(n_seqs):
    if n_seqs<5000:
        return 1
    return 0

def get_zero_thresh(n_seqs):
    return 0.0005*max(1,n_seqs/4000) #min(0.005,max(0.0004,(n_seqs/1400)*0.0005))

def get_beta_ratio(n_seqs):
    return beta_ratio#*max(1,n_seqs/4000)

fams = os.listdir('/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata/pdb')
targets = [f.split('.')[0] for f in fams]
with open(save_path,'w+') as fl:
    for f in targets:
        aln_path = f'~/suppdata/converted_aln/{f}.aln.fasta'
        seq_path = f"~/suppdata/seq/{f}.fasta"
        all_seqs = parse_seqs(f'/mnt/c/Users/mm851/Downloads/interaction_tvgl/suppdata/aln/{f}.aln')
        n_seqs = len(all_seqs)
        if n_seqs>12000:
            continue
        cluster_size = min(max_clust,max(min_clust,len(all_seqs)//clust_denom))
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
                           save_rt = save_rt,
                           beta_ratio= get_beta_ratio(n_seqs),
                           beta_scale=beta_scale,
                           penalty_ty=penalty_ty,
                           all_seq_pos=all_seq_pos,
                           one_lam = get_const_lam(n_seqs),
                           zero_thresh = get_zero_thresh(n_seqs))
        #print(f,cluster_size,len(all_seqs))
        print(f,len(all_seqs))
        fl.write(f'{seq_path} {aln_path} {options} >{save_rt}/logs/{f}_out.log &\n')



