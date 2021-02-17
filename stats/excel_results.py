import sys
import numpy as np
results_file = sys.argv[1]
results = np.load(results_file, allow_pickle=True).item()
#[name,beta,lam,m,M,pr,len(get_seq(seq_path)),i, uid, name, s1, s2]
NAME, BETA, LAMBDA, m, M, PREC, SEQ_LEN, IDX, UID, _, S1, S2= 0,1,2,3,4,5,6,7,8,9,10,11
header = 'name, beta, lambda, min_sep, max_sep, predicted contacts, sequence length, index, uid, result file name, sparsity of corr matrix, sparsity of contact prediction, precision'
header = header.split(',')
all_data = [header]
save_path = sys.argv[2]
for name in results:
    target = name.split('_')[0]
    for res in results[name]:
        print(res)
        for n_contacts in res[PREC]:

            data = list(res)
            data[0]=target
            data[PREC]=n_contacts
            data.append(res[PREC][n_contacts])
            data[M]=res[SEQ_LEN] if res[M] is None else res[M]
            all_data.append(data)

all_data = np.array(all_data,dtype=object)
np.savetxt(save_path, all_data, delimiter=",", fmt='%s')


