import os
import subprocess
import sys
import numpy as np
s,e = int(sys.argv[1]),int(sys.argv[2])
count = 0
with open('args.txt', 'r+') as f:
    for i,x in enumerate(f):
        count+=1

idxs = np.arange(e-s).astype(int)+s
with open('args.txt', 'r+') as f:
    for i,x in enumerate(f):
        if i in idxs:
            #make directory for output
            if '>' in x:
                out_file = x.strip().split('>')[-1]
                if out_file.endswith('&'):
                    out_file=out_file[:-1]
                out_dir = os.path.dirname(out_file)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except:
                    pass

            cmd = f'python /home/mmcpartlon/InteractionTVGL/main.py {x.strip()} '
            print(f'running : {cmd}')
            subprocess.call(cmd, shell = True)