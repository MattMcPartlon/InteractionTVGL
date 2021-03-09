import os
import subprocess
import sys
import time

s,e = int(sys.argv[1]),int(sys.argv[2])
max_running = e-s+1
count = 0
cmds = []
f = 'args.txt'
if len(sys.argv)>3:
    f = sys.argv[3]

if len(sys.argv)>4:
    max_running = int(sys.argv[4])

with open(f, 'r+') as f:
    for i,x in enumerate(f):
        count+=1
        cmds.append(x)

max_running = min(max_running,e-s+1)
e = min(e,count-1)


processes = []
cmds = cmds[s:e+1]
for x in cmds:
    if len(processes)<max_running:
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
        y = x.strip()
        if y.endswith('&'):
            y = y[:-1]
        cmd = f'nohup python /home/mmcpartlon/InteractionTVGL/main.py {y} '
        print(f'running : {cmd}')
        p = subprocess.Popen(cmd, shell = True)
        processes.append(p)
    else:
        print('in else...')
        while True:
            to_rem = []
            for i,p in enumerate(processes):
                if p.poll() is not None:
                    #process finished:
                    to_rem.append(i)
                    print('process finished!')
            if len(to_rem)>0:
                processes = [processes[i] for i in range(len(processes)) if i not in to_rem]
                break
            time.sleep(20)
            print('waiting...')

