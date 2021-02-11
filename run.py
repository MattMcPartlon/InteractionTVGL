import os
import subprocess
import sys
import numpy as np
n_prots = int(sys.argv[1])
count = 0
with open('./args.txt','r+') as f:
    for i,x in enumerate(f):
        count+=1

idxs = np.random.choice(np.arange(count),n_prots,replace=False)
idxs = np.arange(n_prots).astype(int)
with open('./args.txt','r+') as f:
    for i,x in enumerate(f):
        if i in idxs:
            #fam = x.split(' ')[2]
            #save_rt = x.split(' ')[-2]
            cmd = f'python main.py {x.strip()} '
            #if not os.path.exists(f'{save_rt}/logs/'):
            #    os.makedirs(f'{save_rt}/logs/')
            #cmd+= f'>{save_rt}/logs/{fam}.log &'
            subprocess.call(cmd, shell = True)