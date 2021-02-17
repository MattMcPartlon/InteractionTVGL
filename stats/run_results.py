import os
import subprocess
import sys

data_root = sys.argv[1] #pred
ground_truth = sys.argv[2]
seq = sys.argv[3]
save = sys.argv[4]
ty = sys.argv[5]
print('ty == top?',ty=='top')
print(f'{data_root}\n{ground_truth}\n{seq}\n{save}\n')

for data in os.listdir(data_root):
    print('processing :',data)
    if not os.path.isdir(os.path.join(data_root,data)):
        print('not a directory')
        continue
    if not (data.startswith('result') or data.startswith('target')):
        print('not a results folder')
        continue
    if 'output' not in os.listdir(os.path.join(data_root,data)):
        print('no output folder')
        continue
    data_dir = os.path.join(data_root,data,'output')
    if len(os.listdir(data_dir))>15:
        t = '_'+ty if ty == 'top' else ''
        sf = os.path.join(save,data+t+'.npy')
        args = f'{data_dir} {ground_truth} {seq} {sf} {ty}'
        p = subprocess.Popen(f'python ~/InteractionTVGL/process_results.py {args}', shell=True)
        p.wait()
    else:
        print('too few results')