import numpy as np
"""
a = np.random.randint(1,10,(5,10))
print(np.quantile(a,q=(0.5,0.75),axis=0))
;wq:wq
:wq
k = 0
n1 = 38
a_ij = np.empty((n1,n1))
ind = list(range(a_ij.shape[0]))
n = int((-1 + np.sqrt(1 + 8 * a_ij.shape[0])) / 2)
to_rem = []
for i in range(n, 0, -1):
    to_rem.append(k)
    ind.remove(k)
    k = k + i
ind2 = np.arange(n1)
ind2[to_rem]=-1::wq:q:wq
print(ind2[ind2>0])
print(ind)
print(np.array_equal(ind2[ind2>0],ind))


import time
n = int(1e7)
print(n)
start = time.time()
l = list(range(n))
end = time.time()
print(end-start)
fst = end-start
start = time.time()
x = np.arange(n)
end = time.time()
snd = end-start
print(end-start)
print(fst/snd)

print(l[x[5]])

x = np.arange(10)
z = np.arange(10)

x[[1,3,3,4]]+=z[[1,2,3,3]]
print(x)

print(' '.join(['a','b']))
"""

import sys
print(sys.path)
from sklearn.gaussian_process import GaussianProcessRegressor
def gp_reg(xs, ys, target,s=0,e=1):
    gpr = GaussianProcessRegressor(n_restarts_optimizer=3)
    gpr.fit(xs,ys)
    #binary search for target sparsity
    n_samples = 30
    los,his = s*np.ones(len(xs[0])),e*np.ones(len(xs[0]))
    best_xs = np.array(xs[0])
    stop = False
    while not stop:
        test_xs = np.linspace(los,his,n_samples).reshape(n_samples,len(best_xs))
        pred = gpr.predict(test_xs)
        pred = np.abs(pred-target)
        best_idxs = np.argmin(pred,axis=0)
        #print('best idxs',best_idxs)
        assert len(best_idxs)==len(best_xs)
        #print('los, his',los,his)
        best_xs = los + (his-los)*(best_idxs/(n_samples-1))
        #print('best xs',best_xs)
        #print('pred(best xs)-target',gpr.predict([best_xs])-target)

        #reduce the range and continue search around best_xs
        reduced_range = (his-los)*2/5
        los,his = best_xs-reduced_range, best_xs+reduced_range
        #print()
        stop = np.alltrue(np.abs(gpr.predict([best_xs])-target)<1e-4)
        stop = stop or np.alltrue(his[0]-los[0]<1e-5)
    #print('final',best_xs)
    return best_xs
"""
xs = list(np.linspace(-np.pi*np.ones(3),np.pi*np.ones(3),2))
ys = list(np.sin(xs))
for i in range(10):
    best = gp_reg(xs,ys,1,s=-np.pi,e=np.pi)
    xs.append(best)
    ys.append(np.sin(best))
    print()
    print('best xs',best)
    print('best ys',np.sin(best))

"""
a = np.array([0,0,0])
a[[1,1,1]]+=np.array([1,1,1])
print(a)
from typing import Iterable
print(isinstance(a,Iterable))

print(np.matrix(a))

a = [1,2,3,4]
b = list(a)
a.append(5)
print(a)
print(b)


n=1300
import time
start = time.time()
x = np.empty((3,n,n))
x[:]=1/n
a,b,c = x

e = time.time()-start
print(a.shape)
print(b.shape)
print(c.shape)
print(e)
start = time.time()
a = np.ones((n,n))/n
b = np.copy(a)
c = np.copy(a)
z = time.time()-start
print(z)
print(e/z)


a = [[1,1,1],[2,2,2]]
b = [[1,1,1],[2,2,2]]
b = [[]]*len(a)
c = np.concatenate((a,b),axis=1)
print(c)


from utils.utils import adjust_lam_and_beta
lam = [1e-3]*4
beta = [1e-3]*3
prev_sps = np.random.uniform(0,0.06,(4,4))
prev_betas =  np.random.uniform(0,0.06,(4,3))
prev_lams =  np.random.uniform(1e-6,1e-3,(4,4))
theta_set = [np.random.uniform(1e-7,1,(20,20)) for _ in range(4)]
print(adjust_lam_and_beta(theta_set,lam,beta,prev_sps,prev_lams,prev_betas,0.03, sep = 0))
if 1 :
    print('this works')


"""
previous lambs : [[0.002, 0.002, 0.002, 0.002], [0.004, 0.004, 0.004, 0.004], [0.002, 0.002, 0.002, 0.002], [1e-05, 0.007079958751572829, 0.0024899790002237375, 0.005710210419643475], [0.005781852832196007, 0.005515772484379949, 0.0007298086356535523, 0.003437330282969413], [0.003758766969565116, 0.004423477381878222, 0.003913358233397856, 0.0034562802598695876], [0.0037665330501864243, 0.004120124060657059, 0.003697168351078303, 0.00389310716807552], [0.0036805651125704666, 0.004117926079428954, 0.003621625110340105, 0.003888640877512781]]
previous betas : [[0.00025, 0.00025, 0.00025], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]]]
sparsities : [[0.04598088025302333, 0.07083870976445329, 0.06058948393680892, 0.07061499626881863], [0.015622621638873357, 0.02919135103390485, 0.023077564729082956, 0.02721367876495761], [0.04598088025302333, 0.07083870976445329, 0.06058948393680892, 0.07061499626881863], [0.31752814349500735, 0.013280262646389006, 0.042889137259583876, 0.015294808295521255], [0.008264133915123324, 0.01855562922737003, 0.12734019492977533, 0.03628183223826139], [0.017108124217424334, 0.025441508139236582, 0.023902044516512417, 0.03383379953427122], [0.017163659125385904, 0.028165416681562163, 0.025928731400583632, 0.02844938667652355], [0.01609545528844089, 0.026056664042810894, 0.024723151738681053, 0.026447881613066486]]
current lamb : [0.003597418070041242, 0.0040858193205906106, 0.0035473979164839047, 0.003861268954311016]
current beta : [[0.00025], [0.00025], [0.00025]]
"""
previous_lambs = [[0.002, 0.002, 0.002, 0.002], [0.004, 0.004, 0.004, 0.004], [0.002, 0.002, 0.002, 0.002], [1e-05, 0.007079958751572829, 0.0024899790002237375, 0.005710210419643475], [0.005781852832196007, 0.005515772484379949, 0.0007298086356535523, 0.003437330282969413], [0.003758766969565116, 0.004423477381878222, 0.003913358233397856, 0.0034562802598695876], [0.0037665330501864243, 0.004120124060657059, 0.003697168351078303, 0.00389310716807552], [0.0036805651125704666, 0.004117926079428954, 0.003621625110340105, 0.003888640877512781]]
previous_betas = [[0.00025, 0.00025, 0.00025], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]], [[0.00025], [0.00025], [0.00025]]]
sparsities = [[0.04598088025302333, 0.07083870976445329, 0.06058948393680892, 0.07061499626881863], [0.015622621638873357, 0.02919135103390485, 0.023077564729082956, 0.02721367876495761], [0.04598088025302333, 0.07083870976445329, 0.06058948393680892, 0.07061499626881863], [0.31752814349500735, 0.013280262646389006, 0.042889137259583876, 0.015294808295521255], [0.008264133915123324, 0.01855562922737003, 0.12734019492977533, 0.03628183223826139], [0.017108124217424334, 0.025441508139236582, 0.023902044516512417, 0.03383379953427122], [0.017163659125385904, 0.028165416681562163, 0.025928731400583632, 0.02844938667652355], [0.01609545528844089, 0.026056664042810894, 0.024723151738681053, 0.026447881613066486]]
current_lamb = [0.003597418070041242, 0.0040858193205906106, 0.0035473979164839047, 0.003861268954311016]
current_beta = [[0.00025], [0.00025], [0.00025]]
print('gp reg')

def gp_reg(xs, ys, target, s=1e-7, e=1e-2, max_tries = 100):
    gpr = GaussianProcessRegressor(n_restarts_optimizer=3)
    gpr.fit(xs, ys)
    return reduce_to_convergence(gpr,xs[-1],target,s,e,max_tries)

def reduce_range(gpr, curr_xs, curr_hi, curr_lo, target, s, e):
    tmp = list(curr_xs)
    lbs,ubs = [],[]
    for i in range(len(curr_xs)):
        l,h = curr_lo[i],curr_hi[i]
        xl,xh = l,h
        for x in np.linspace(l,h,40):
            tmp[i]=x
            if gpr.predict([tmp]).ravel()[i]>target:
                xl = max(x,xl)
            if gpr.predict([tmp]).ravel()[i]<target:
                xh = min(xh,x)
        lbs.append(xl)
        ubs.append(xh)
        tmp[i]=curr_xs[i]
    assert np.alltrue(np.array(ubs)<=curr_hi)
    assert np.alltrue(np.array(lbs) >= curr_lo)
    rng = (curr_hi-curr_lo)*2/5
    for i in range(len(lbs)):
        lbs[i] = min(lbs[i],curr_xs[i]-rng[i])
        ubs[i] = max(ubs[i], curr_xs[i] + rng[i])
    return lbs,ubs

def reduce_to_convergence(gpr, _curr_xs,target, s, e, max_tries = 100):
    curr_xs = list(_curr_xs)
    prev_xs = list(curr_xs)
    new_xs = list(curr_xs)
    stop = False
    curr_lo,curr_hi = s*np.ones(len(curr_xs)),e*np.ones(len(curr_xs))
    for c in range(max_tries):
        for i in range(len(curr_xs)):
            best ,bx = 1, curr_xs[i]
            l,h = curr_lo[i],curr_hi[i]
            curr_x = curr_xs[i]
            for x in np.linspace(l,h,40):
                curr_xs[i]=x
                si = gpr.predict([curr_xs]).ravel()[i]
                if abs(si-target)<best:
                    bx = x
                    best = abs(si-target)
            curr_xs[i]=curr_x
            new_xs[i]=bx

        prev_xs = list(curr_xs)
        curr_xs = list(new_xs)

        rng = np.abs(np.array(curr_xs)-np.array(prev_xs))
        curr_lo = np.array(curr_xs)-rng
        curr_hi = np.array(curr_xs)+rng
        curr_lo=np.maximum(curr_lo,s)
        curr_hi=np.maximum(curr_hi,e)
        stop = np.max(np.abs(curr_hi-curr_lo))<1e-6
        stop = stop or np.max(np.abs(gpr.predict([new_xs])-target))<0.5*1e-3
        print(np.round(curr_xs,6),np.round(gpr.predict([curr_xs]),5))
        if stop:
            break

    print(c)
    return curr_xs, gpr.predict([curr_xs])







def simple_gpr_inverse(xs,ys,target,s=1e-6,e=1e-2):
    gpr = GaussianProcessRegressor(n_restarts_optimizer=3)
    gpr.fit(ys,xs)
    lams = gpr.predict([[target]*len(xs[0])]).ravel()
    return np.maximum(s,np.minimum(lams,e))






for i in range(len(previous_lambs)-2):
    l,s = previous_lambs[:i+2], sparsities[:i+2]
    print('xs',np.round(l,4))
    print('ys',np.round(s,4))
    #l,pred = gp_reg(l,s,0.03)

    #print('output',np.round(l,6),np.round(pred,4))
    print(simple_gpr_inverse(l,s,0.03))
    print()

from random import shuffle
a = [1,2,3,4,5,6]
shuffle(a)
print(a)

x = ''
y = x or 'hello'
print(y)

x = 'hi {}'
print(x.format([1,2,3]))

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

print(ncr(10,5)*ncr(5,3))