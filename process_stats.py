import numpy as np
from collections import defaultdict
from itertools import product
sparsities = [(0.02,1),(0.02,0.5),(0,1),(0.02,0.07)]
top = False
root = '/mnt/c/Users/mm851/Downloads'
suff = '_top_L' if top else '_top_N'
prec_header = 'method,cluster_type,min_sep,1,2,5,10,25,50,100,ls,us'.split(',')
overall_save_path = root+f'/precision_results/precition_avgs_by_method{suff}.csv'
if top:
    prec_header = 'method,cluster_type,min_sep,L,L/2,L/5,L/10,ls,us'.split(',')
overall_data = [prec_header]
for ls,us in sparsities:
    header ='method,target,seq len, msa depth,cluster type,s1,s2, min sep, max sep, data type, 1,2,5,10,25,50,100'.split(',')
    key_map = {x.strip() : i for i,x in enumerate(header)}
    s_key=key_map['s2']
    path = root+f'/precision_results/precision{suff}.csv'
    print(path)
    save_path = root+f'/precision_results/precition_avgs_by_method{suff}_{ls}_{us}.csv'
    data = np.loadtxt(path,delimiter=",",dtype = object)
    data_keys = [-7,-6,-5,-4,-3,-2,-1]
    if top:
        data_keys = [-4,-3,-2,-1]

    def get_entry_key(d):
        return d[key_map['min sep']], d[key_map['cluster type']], d[key_map['data type']]
    #construct a map to filter data with
    data_map = defaultdict(lambda : defaultdict(list))
    for d in data:
        method,  = d[key_map['method']],
        min_sep, cluster_type, data_type = get_entry_key(d)
        data_map[method][(min_sep,cluster_type,data_type)].append(np.array([str(x) for x in d]))


    #top scoring prediction rank by average precision for top n predicted contacts
    # seps 5-9, 10-23, >23

    #gather this data for each method
    prec_data = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    cluster_types = ['FIRST','ALL','NOT FIRST OR TARGET','TARGET']
    min_seps = ['5','10','23']
    if top:
        min_seps = ['5','9','12','24']

    for method in data_map:
        for min_sep,cluster_type in product(min_seps,cluster_types):
            key = (min_sep,cluster_type,'0.75')
            prec_scores = []
            for data in data_map[method][key]:
                if float(data[s_key])>ls and float(data[s_key])<us:
                    print(data[data_keys])
                    prec_scores.append(data[data_keys].astype(float))
            if len(prec_scores)>0:
                means = np.mean(prec_scores,axis=0)
                prec_data[method][cluster_type][min_sep]=means

    #all precision scores for each cluster type/ method/ sep are in prec_data
    #flatten out the map and write to a file
    method_name_map = {
        'results_4':'All seqs + largest cd hit clusters (n_cov = 4)',
        'results_5_same_corr_mat': 'psicov high penalty (n_cov = 1)',
        'results_6_small_clust': 'All seqs + seq similarity >0.4 with target (n_cov = 2)',
        'results_phylo_low_pen_more_iters':'All seqs + partition into 3 clusters by phylo- low penalty (n_cov = 4)',
        'results_phylo_no_shrink_n_scale_more_iters_high_pen':'All seqs + partition into 3 clusters by phylo- high penalty (n_cov = 4)',
        'results_psi_cov': 'psicov low penalty (n_cov = 1)',
        'results_random_partition_more_iters_high_pen':'random partition of sequences (n_cov = 4)',
        'results_phylo_no_shrink_n_scale_more_iters_partition_only_high_pen':'only phylo clusters (cluster of all seq removed)- high penalty (n_cov = 3)'
    }

    prec_header = 'method,cluster_type,min_sep,1,2,5,10,25,50,100,ls,us'.split(',')
    if top:
        prec_header = 'method,cluster_type,min_sep,L,L/2,L/5,L/10,ls,us'.split(',')
    data_to_write = [prec_header]
    for method in prec_data:
        if method not in method_name_map:
            continue
        for cluster_type in prec_data[method]:
            for min_sep in prec_data[method][cluster_type]:
                print('min seps',[prec_data[method][cluster_type].keys()])
                d = [method_name_map[method], cluster_type, min_sep]
                if isinstance(prec_data[method][cluster_type][min_sep],np.ndarray):
                    for v in prec_data[method][cluster_type][min_sep]:
                        d.append(v)
                    d.append(ls)
                    d.append(us)
                    data_to_write.append(d)
    data_to_write = np.array(data_to_write,dtype=object)
    np.savetxt(save_path, data_to_write, delimiter=",", fmt='%s')


    #mean precision values for L, L/2, L/5, L/10
    #sep  [i-j]>4, [i-j]>8, [i-j]>11, [i-j]>23

    #same for criteria "any heavy atom <6A"

    #correlation of precision (Top L, Top L/2) to alignment length (weighted and unweighted)

    #average precision for different sparsities values use (0,0.02), (0.02,0.04),(0.04,0.10),(0.10,0.30),(>0.30)
    overall_data.extend(data_to_write[1:])

overall_data = np.array(overall_data,dtype=object)
np.savetxt(overall_save_path, overall_data, delimiter=",", fmt='%s')