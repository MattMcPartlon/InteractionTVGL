import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

data_path = './results_4_precision.npy'
save_path = './results_4'
if len(sys.argv)>1:
    data_path = sys.argv[1]
    save_path = sys.argv[2]

#target : [beta,lam,m,M,pr,seq, idx, uid]
NAME, BETA, LAMBDA, m, M, PREC, SEQ_LEN, IDX, UID, _, S1, S2, TARGET_CLUST= 0,1,2,3,4,5,6,7,8,9,10,11,12
pr_counts = [1,2,5,10,25,50,100]
data = np.load(data_path,allow_pickle=True).item()


def get_by_type(data, type : int, val, scale_to_seq_len = True):
    """

    :param data:
    :param type:
    :param val:
    :param scale_to_seq_len:
    :return:
    """
    plot_data = defaultdict(lambda:defaultdict(list))
    count = 0
    for name in data:
        target_name = name.split('_')[0]
        count+=1
        target_dat = data[name]
        for entry in target_dat:
            ty = entry[type]
            if scale_to_seq_len:
                ty*=entry[SEQ_LEN]
            plot_data[entry[m]]['x'].append(ty)
            for k,v in entry[PREC].items():
                plot_data[entry[m]][k].append(v)
    return plot_data

def get_quantiles(data, idx = None):
    plot_data = defaultdict(lambda :defaultdict(dict))
    count = 0
    for name in data:
        target_name = name.split('_')[0]
        count+=1
        target_dat = data[name]
        #print(target_dat)
        for entry in target_dat:
            plot_data[target_name][entry[m]]['x']=pr_counts
            if idx is None or entry[IDX] == idx:
                for k,v in entry[PREC].items():
                    if k not in plot_data[target_name][entry[m]]:
                        plot_data[target_name][entry[m]][k]=[]
                    plot_data[target_name][entry[m]][k].append(v)
    #replace data with quantile info
    #print(plot_data)
    temp = {}
    for target in plot_data:
        temp[target]={}
        for sep in plot_data[target]:
            temp[target][sep]={}
            d = plot_data[target][sep]
            for count in d:
                if count =='x':
                    temp[target][sep]['x']=d[count]
                    continue
                #get quantiles
                data = np.quantile(d[count],q=(0,0.25,0.5,0.75,1))
                temp[target][sep][count]=data
    return temp

def get_avgs(data, idx=None):
    qts = get_quantiles(data, idx = idx)
    #print(qts)
    #print()
    avgs = defaultdict(lambda:defaultdict(list))
    for target in qts:
        for sep in qts[target]:
            d = qts[target][sep]
            for count in qts[target][sep]:
                if count =='x':
                    continue
                avgs[sep][count].append(d[count][-2])
    #print(avgs)
    #take avg and place all in same list
    plot_dat = {'x':pr_counts}
    for sep in avgs:
        plot_dat[sep]=[]
        for count in pr_counts:
            plot_dat[sep].append(np.mean(avgs[sep][count]))
    return plot_dat



def bucket_tys(xs,ys, n_buckets=6):
    mn,mx = np.min(xs),np.max(xs)

    rng = (mx-mn)/n_buckets + 1e-8
    new_xs = np.linspace(mn, mx, n_buckets)
    new_ys = defaultdict(list)
    for x,y in zip(xs,ys):
        idx = int((x-mn)//rng)
        new_ys[idx].append(y)
    idxs = [i for i in new_ys if len(new_ys[i])>1]
    idxs = sorted(idxs)
    stds = [np.std(new_ys[i]) for i in idxs]
    new_xs = [new_xs[i] for i in idxs]
    new_ys = [np.mean(new_ys[idx]) for idx in idxs]
    return new_xs,new_ys,stds

pd = get_avgs(data)

cols = len(pd)-1
rows = 1
fig = plt.figure(figsize=(10,5))
fig.tight_layout()
axs = fig.subplots(rows,cols).flat
ax_idx = -1
for sep in pd:
    if sep == 'x':
        continue
    ax_idx+=1
    ax = axs[ax_idx]
    ax.set_title(f'SEP = {sep}')
    x, y = pd['x'], pd[sep]
    ax.plot(x,y,'o')

fig.tight_layout()
fig.savefig(save_path+'_avgs_idx_all.jpeg')

pd = get_avgs(data, idx = 0)

cols = len(pd)-1
rows = 1
fig = plt.figure(figsize=(10,5))
fig.tight_layout()
axs = fig.subplots(rows,cols).flat
ax_idx = -1
for sep in pd:
    if sep == 'x':
        continue
    ax_idx+=1
    ax = axs[ax_idx]
    ax.set_title(f'SEP = {sep}')
    x, y = pd['x'], pd[sep]
    ax.plot(x,y,'o')

fig.tight_layout()
fig.savefig(save_path+'_avgs_idx_0.jpeg')

pd = get_avgs(data, idx=1)

cols = len(pd)-1
rows = 1
fig = plt.figure(figsize=(10,5))
fig.tight_layout()
axs = fig.subplots(rows,cols).flat
ax_idx = -1
for sep in pd:
    if sep == 'x':
        continue
    ax_idx+=1
    ax = axs[ax_idx]
    ax.set_title(f'SEP = {sep}')
    x, y = pd['x'], pd[sep]
    ax.plot(x,y,'o')

fig.tight_layout()
fig.savefig(save_path+'_avgs_idx_1.jpeg')



pd = view_by_type(data, BETA, scale_to_seq_len = True)

cols = len(pd)
rows = len(pr_counts)
fig = plt.figure(figsize=(10,10))
fig.tight_layout()
axs = fig.subplots(rows,cols).flat
ax_idx = 0
n_buckets = 10
for pr_thresh in pr_counts:
    for sep in pd:
        ax = axs[ax_idx]
        ax.set_title(f'SEP = {sep}, n_contacts = {pr_thresh}')
        x,y = pd[sep]['x'], pd[sep][pr_thresh]
        x,y,err = bucket_tys(x,y, n_buckets=n_buckets)
        ax.errorbar(x,y,yerr=err,fmt='')
        ax_idx+=1
fig.tight_layout()
fig.savefig(save_path+'_by_beta_scaled.jpeg')

pd = view_by_type(data, BETA, scale_to_seq_len = False)

cols = len(pd)
rows = len(pr_counts)
fig = plt.figure(figsize=(10,10))
fig.tight_layout()
axs = fig.subplots(rows,cols).flat
ax_idx = 0
n_buckets = 10
for pr_thresh in pr_counts:
    for sep in pd:
        ax = axs[ax_idx]
        ax.set_title(f'SEP = {sep}, n_contacts = {pr_thresh}')
        x,y = pd[sep]['x'], pd[sep][pr_thresh]
        x,y,err = bucket_tys(x,y, n_buckets=n_buckets)
        ax.errorbar(x,y,yerr=err,fmt='')
        ax_idx+=1
fig.tight_layout()
fig.savefig(save_path+'_by_beta_unscaled.jpeg')













