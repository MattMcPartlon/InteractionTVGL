import numpy as np
def parse_psicov(path, start, end):
    in_prec = False
    prec = []
    with open(path,'r+') as f:
        for line in f:
            if line.strip().startswith(start):
                in_prec = True
                continue
            if line.strip().startswith(end):
                return np.array(prec)
            if in_prec:
                data = [float(x) for x in line.strip().split(' ')]
                prec.append(data)
    return None

def parse_precision_mat(path):
    return parse_psicov(path,start = 'precision matrix',end ='end of precision matrix')

def parse_cov_mat(path):
    return parse_psicov(path, start='covariance matrix', end='end of covariance matrix')

import os
import sys
for x in os.listdir(sys.argv[1]):
    name = x.split('.')[0]
    path = os.path.join(sys.argv[1],x)
    save_pth = os.path.join(sys.argv[2],name+'.npy')
    data = {}
    data['theta set']=[parse_precision_mat(path)]
    np.save(save_pth,data)