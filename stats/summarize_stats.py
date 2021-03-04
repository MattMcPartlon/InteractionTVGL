from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

from utils.utils import load_npy

# compare first clust to target clust
excel_path = '/mnt/c/Users/mm851/Desktop/prec_results2.xlsx'
meff_path = '/mnt/c/Users/mm851/Desktop/meffs.npy'
sheet_names = ['TOP_L__final']
name_map = {'results_psicov_22': 'psicov',
            'results_target_sp_03_larger_clust_size_2_clust_min_beta_ratio_025': 'TVGL_phylo_b_025_sp_03',
            'results_target_sp_032_beta_0005_fix_phylo_gpr_inv': 'TVGL_phylo_b_0005_sp_032',
            'results_target_sp_032_beta_001_fix_phylo_gpr_inv': 'TVGL_phylo_b_001_sp_032',
            'results_target_sp_032_beta_001_fix_phylo_gpr_inv_first_only': 'TVGL_phylo_b_001_sp_032_F',
            'results_target_sp_032_beta_0005_fix_phylo_gpr_inv_first_only': 'TVGL_phylo_b_0005_sp_032_F',
            'results_target_sp_03_beta_larger_8_fix_phylo_gpr_inv_weighted_by_clust': 'TVGL_clust_weighted_low_beta',
            'results_target_sp_03_beta_0005_fix_phylo_gpr_inv_weighted_by_clust_curve_fit':
                'TVGL_clust_weighted_high_beta',
            }
del name_map['results_target_sp_03_larger_clust_size_2_clust_min_beta_ratio_025']

methods_to_compare = set(name_map.keys())
excel_data = pd.read_excel(excel_path, sheet_name=sheet_names)
meff_map = load_npy(path=meff_path)

target_list = '1a3aA	1a6mA	1aapA	1abaA	1atlA	1atzA	1avsA	1bdoA	1bebA	1behA	1bkrA	1brfA	1c44A	1c52A	1c9oA	1cc8A	1chdA	1ckeA	1ctfA	1cxyA	1cznA	1d0qA	1d1qA	1dbxA'
targets = set(target_list.split('\t'))
for key in excel_data:
    print(key)


def filter_name(name):
    if name in name_map:
        return name_map[name]
    return name


def method_filter(method):
    return method in methods_to_compare


def target_filter(target):
    return target in targets


def get_methods():
    return list(methods_to_compare)


# plot average precision against meff
#    for each method
#         For each sep
#             for TopL, TopL/2, TopL/5

def get_prec_vs_meff(sheet_data, prec, sep, data_type=1, cluster_type='FIRST'):
    data = get_method_precs(sheet_data, prec, sep, data_type, cluster_type)
    labels = []
    xs, ys = [], []
    for method in get_methods():
        method_x, method_y = [], []
        if method_filter(method):
            labels.append(filter_name(method))
            for target in data[method]:
                if target_filter(target):
                    method_x.append(meff_map[target])
                    method_y.append(data[method][target])
            xs.append(method_x)
            ys.append(method_y)
    return xs, ys, labels


def plot_prec_vs_meff(n_rows, n_cols, xs, ys, labels, title, save_path='./test.png'):
    fig_kwargs = {'figsize': (n_cols * 4, n_rows * 4)}
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, **fig_kwargs)
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("meff")
    plt.ylabel("target precision")
    ax = ax.flatten()
    for i, a in enumerate(ax):
        a.scatter(xs[i], ys[i])
        r = corr(xs[i], ys[i])
        a.set_title(f"{labels[i]}, r = {r}")

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9], h_pad=0.1, w_pad=0.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9], h_pad=0.1, w_pad=0.1)
    plt.savefig(save_path, bbox_inches='tight',
                dpi=150)


def corr(x, y):
    x, y = np.array(x), np.array(y)
    return np.round(np.corrcoef(x, y)[0, 1], 3)


def get_method_precs(data: pd.DataFrame, pr, sep, data_type, cluster_type):
    """
    returns a dict of form dict[method]->dict[target]->prec
    :param pr:
    :param sep:
    :param data_type:
    :return:
    """
    pr_data = defaultdict(lambda: defaultdict(lambda: 0))
    for _, row in data.iterrows():
        if not row['min sep'] == sep:
            continue
        if not row['cluster type'] == cluster_type:
            continue
        if not row['data type'] == data_type:
            continue
        method = row['method']
        target = row['target']
        val = row[pr]
        pr_data[method][target] = val
    return pr_data


# 4 tables with methods as rows, average precision for top L,L/2,L/5,L/10 as cols
def get_top_L_precision_table(sheet_data, data_type=1, cluster_type='FIRST', precs_labels=['L', 'L/2', 'L/5', 'L/10']):
    table_data = []
    titles = []
    row_names = []
    for sep in [5, 9, 12, 24]:
        titles.append(str(sep))
        sep_data = []
        points = {}
        for l in precs_labels:
            points[l] = get_method_precs(sheet_data, l, sep, data_type, cluster_type)
        _row_names = []
        for method in get_methods():
            row = []
            if method_filter(method):
                _row_names.append(filter_name(method))
                method_precs = [[points[l][method][t] for l in precs_labels] for t in points[precs_labels[0]][method] if
                                target_filter(t)]
                method_precs = np.mean(method_precs, axis=0)
                row.extend(p for p in np.round(method_precs, 3))
            if method_filter(method):
                sep_data.append(row)
        table_data.append(sep_data)
        row_names.append(_row_names)
    # returns table titles, column labels, row labels, table data
    return titles, row_names, [precs_labels] * len(table_data), table_data


def plot_table(table_data, n_rows, n_cols, title='', table_names=None, row_labels=None, col_labels=None,
               save_path='./test.png'):
    fig_kwargs = {}
    # fig_kwargs['figsize']=(12,12)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, **fig_kwargs)
    ax = ax.flatten()
    for i, a in enumerate(ax):
        kwargs = {}
        row_offset, col_offset = 0, 0
        if i % n_cols == 0:
            kwargs['rowLabels'] = row_labels[i]
            # col_offset = 1
        if i < n_cols:
            row_offset = 1
            kwargs['colLabels'] = col_labels[i]
        table = a.table(table_data[i],
                        loc="center",
                        cellLoc='center',
                        **kwargs)
        table.PAD = 0.2
        a.axis("off")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.auto_set_column_width(col=list(range(len(table_data[i][0]))))
        if table_names is not None:
            a.set_title(table_names[i])
        # highlight best method for each top L
        best_methods = np.argmax(table_data[i], axis=0)
        assert len(best_methods) == len(table_data[i][0])
        for j, r in enumerate(best_methods):
            table.get_celld()[(row_offset + r, col_offset + j)].set_text_props(
                fontproperties=FontProperties(weight='bold', size=7))

    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(top=0.9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9], h_pad=0.1, w_pad=0.1)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.9],h_pad = 0.1, w_pad=0.1)
    # plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight',
                dpi=150)


def get_prec_vs_meff_bins(sheet_data, prec, sep, data_type=1, cluster_type='FIRST', n_bins=7):
    xs, ys, labels = get_prec_vs_meff(sheet_data, prec, sep, data_type=data_type, cluster_type=cluster_type)
    # xs correspond to meff
    print(labels)
    all_data = []
    bin_labels = []
    for j, y in enumerate(np.array(ys)):
        method_data = []
        order = np.argsort(xs[j])
        chunk = len(xs[j]) // n_bins
        srtd_meffs = np.array(xs[j])[order]
        srtd_meffs = np.round(srtd_meffs, 1)
        s, e = 0, chunk
        for i in range(n_bins):
            if i < len(xs[0]) % n_bins:
                e += 1
            bin_labels.append(f"{srtd_meffs[s]}-{srtd_meffs[e - 1]}")
            method_data.append(np.mean(y[order[s:e]]))
            s = e
            e = e + chunk
        all_data.append(method_data)
    return all_data, bin_labels[:n_bins], labels


def plot_prec_vs_meff_bins(data, bin_labels, methods, title='', save_path='./test.png'):
    n_groups = len(data[0])
    print(methods)
    plt.figure(figsize=(len(data[0]) * len(data) / 2.5, 6))
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    rects = []
    for i, m in enumerate(methods):
        locs = index + (i * bar_width)
        r = plt.bar(locs,
                    data[i],
                    bar_width,
                    alpha=opacity,
                    label=m
                    )
        rects.append(r)
        autolabel(r, plt, data[i])

    plt.xlabel('meff (min-max)')
    plt.ylabel('average precision')
    plt.title(title)
    plt.xticks(index + bar_width, bin_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def autolabel(rects, plt, label):
    """
    Attach a text label above each bar displaying its height
    """
    for i, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 str(np.round(label[i], 2)),
                 ha='center', va='bottom', size=7.5)


def filter_sheet(sheet, criteria):
    """
    returns a dict of form dict[method]->dict[target]->prec
    :param pr:
    :param sep:
    :param data_type:
    :return:
    """
    data = []
    for _, row in sheet.iterrows():
        keep = True
        for k, v in criteria.items():
            if str(row[k]) != str(v):
                keep = False
                break
        if keep:
            data.append(row)
    return data


excel_sheet = excel_data['TOP_L__final']

for cluster_type in ['FIRST', 'AVG']:
    table_names = [f"|i-j|>{sep}" for sep in [4, 8, 11, 23]]
    titles, row_names, col_names, table_data = get_top_L_precision_table(excel_sheet, cluster_type=cluster_type)
    title = f'Top L/k precision by Minimum Sequence Separation\n For first 40 protein families\n (k=1,2,5,8), cluster_type = {cluster_type}'
    plot_table(table_data, 2, 2, title, table_names, row_names, col_names,
               save_path=f'plots/prec_avgs_{cluster_type}.png')

    xs, ys, labels = get_prec_vs_meff(excel_sheet, 'L', 5, cluster_type=cluster_type)
    n_rows = 1
    n_cols = len(get_methods())
    title = f'Comparison of Top L precision vs. meff\n for cluster_type = {cluster_type}'
    plot_prec_vs_meff(3, 2, xs, ys, labels, title, save_path=f'plots/meff_{cluster_type}.png')

    # top scoring prediction length by sep (plot 1,2,5,...)
    # top scoring prediction by
    title = f'Top L Precision binned by meff\n cluster_type = {cluster_type}'
    # average precision by meff (meff_ranges s.t. ~5 targets in each each range)
    all_data, bin_labels, methods = get_prec_vs_meff_bins(excel_sheet, 'L', 5, cluster_type=cluster_type, n_bins=5)
    plot_prec_vs_meff_bins(all_data, bin_labels, methods, title=title,
                           save_path=f'plots/prec_vs_meff_binned_{cluster_type}.png')


# make some sheets that are easy to review
# hide the data type#

# do this for top L and 1,2,5,10,25,50,100
# Sheet 1 - all data
# method, target, meff, seq len, cluster_type, min_sep, max_sep, precision_type, precision_value
def get_method_and_target_tables(sheet_map, data_type=0.75):
    entries = ['method', 'target', 'seq len', 'msa depth', 'meff', 'cluster type', 'min sep', 'max sep']
    header = 'method,target,seq len,msa depth,meff,cluster_type,min_sep,max_sep,precision_type,precision_value'
    header = header.split(',')
    sheet_data = defaultdict(list)
    for sheet_name in sheet_map:
        sheet = sheet_map[sheet_name]
        if 'top_l' in sheet_name.lower():
            prs = ['L/10', 'L/5', 'L/2', 'L']
        else:
            prs = '1,2,5,10,25,50,100'.split(',')
        filtered_sheet = filter_sheet(sheet, {'data type': data_type})
        # get the values per method and target
        for row in filtered_sheet:
            method = row['method']
            if method_filter(row['method']):
                if target_filter(row['target']):
                    _row_data = [filter_name(method)] + [row[entry] for entry in entries[1:]]
                    for pr in prs:
                        row_data = list(_row_data)
                        row_data.append(pr)
                        row_data.append(row[pr])
                        sheet_data[sheet_name].append(row_data)
        return header, sheet_data


from functools import partial


def nested_dd(max_depth, constructor):
    if max_depth > 1:
        return defaultdict(partial(nested_dd, max_depth - 1, constructor))
    return defaultdict(constructor)




def get_method_avg_tables(sheet_map, data_type = 0.75):
    entries = ['method','cluster type','min sep','max sep']
    header = 'method,cluster_type,min_sep,max_sep,precision_type,precision_value'
    header = header.split(',')
    data_map = defaultdict(list)
    for sheet_name in sheet_map:
        sheet = sheet_map[sheet_name]
        if 'top_l' in sheet_name.lower():
            prs = ['L/10','L/5','L/2','L']
        else:
            prs = '1,2,5,10,25,50,100'.split(',')
        filtered_sheet = filter_sheet(sheet,{'data type':data_type})
        #get the averages per method and target
        method_data = nested_dd(5,list)
        for row in filtered_sheet:
            method = row['method']
            if method_filter(row['method']):
                if target_filter(row['target']):
                    [meth,ct,minsep,maxsep] = [filter_name(method)]+[row[entry] for entry in entries[1:]]
                    if str(maxsep)=='nan':
                        maxsep = 'L'
                    for pr in prs:
                        method_data[meth][ct][minsep][maxsep][pr].append(row[pr])
        for method in method_data:
            print('in avg, processing method :',filter_name(method),'sheet :',sheet_name)
            dat_a = method_data[method]
            ra = [method]
            #print('methods',method_data.keys())
            for cluster_ty in dat_a:
                rb = ra +[cluster_ty]
                dat_b = dat_a[cluster_ty]
                #print('clusters', dat_a.keys())
                for min_sep in dat_b:
                    rc = rb + [min_sep]
                    dat_c = dat_b[min_sep]
                    #print('min seps', dat_b.keys())
                    for maxsep in dat_c:
                        dat_d = dat_c[maxsep]
                        rd = rc + [maxsep]
                        #print('max seps', dat_c.keys())
                        for pr in dat_d:
                            #print(dat_d)
                            #exit(1)
                            dat_e = dat_d[pr]
                            re = rd + [pr]
                            re.append(np.mean(dat_e))
                            data_map[sheet_name].append(re)
    return header, data_map



#save precision_by_target
save_f1 = '/mnt/c/Users/mm851/Desktop/precision_summary_draft_2.xlsx'

h1,data_map1 = get_method_and_target_tables(excel_data)
h2,data_map2 = get_method_avg_tables(excel_data)
data1, data2 = [], []
for k,v in data_map1.items():
    data1.extend(v)
for k, v in data_map2.items():
    data2.extend(v)

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
    
with pd.ExcelWriter(save_f1, mode='w+') as writer:
    df1.to_excel(writer, sheet_name='method_by_target', header=h1, float_format="%.5f")
    df2.to_excel(writer, sheet_name='method_avgs', header=h2, float_format="%.5f")


