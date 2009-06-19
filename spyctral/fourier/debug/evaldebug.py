import numpy as np
import scipy as sp
from scipy import rand, randn
from numpy.linalg import norm
import fourier as gf
import spyctral as sp

import spyctral.fourier as fs

N = 100
gamma = 0.
delta = 0.
L = 3.
a = 5.

keys1 = ['gamma', 'delta', 'scale','shift']
keys2 = ['g','d','scale','shift']

params = dict(zip(keys1, [gamma,delta,L,a]))
params_old = dict(zip(keys2, [gamma,delta,L,a]))

def shiftstuff():
    gamma = 10*rand()
    delta = 10*rand()
    L = 5*rand()
    a = 5*randn()

    return [dict(zip(keys1,[gamma,delta,L,a])),\
            dict(zip(keys2,[gamma,delta,L,a]))]

[params,params_old] = shiftstuff()

[x_ref,w_ref] = gf.quad.genfourier_gquad(N,**params_old)
[x,w] = fs.quad.gq(N,**params)

print "GQ error", norm(x_ref-x) + norm(w_ref - w)

[x_ref,w_ref] = gf.quad.genfourierw_pgquad(N,**params_old)
[x,w] = fs.quad.pgq(N,**params)

print "PGQ error", norm(x_ref-x) + norm(w_ref - w)

ks = sp.common.indexing.integer_range(N)
funs_ref = gf.genfourier.genfourier(x,ks,**params_old)
funs = fs.eval.fseries(x,ks,**params)

print "Unweighted function error", norm(funs-funs_ref)

funs_ref = gf.genfourier.genfourierw(x,ks,**params_old)
funs = fs.eval.weighted_fseries(x,ks,**params)

print "Weighted function error", norm(funs-funs_ref)

dfuns_ref = gf.genfourier.dgenfourier(x,ks,**params_old)
dfuns = fs.eval.dfseries(x,ks,**params)

print "Derivative unweighted function error", norm(dfuns-dfuns_ref)

dfuns_ref = gf.genfourier.dgenfourierw(x,ks,**params_old)
dfuns = fs.eval.dweighted_fseries(x,ks,**params)

print "Derivative weighted function error", norm(dfuns-dfuns_ref)
