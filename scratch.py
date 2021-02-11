import numpy as np
"""
a = np.random.randint(1,10,(5,10))
print(np.quantile(a,q=(0.5,0.75),axis=0))

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
ind2[to_rem]=-1
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