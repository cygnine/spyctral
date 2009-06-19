# Fourier debugging of fft

import spyctral
import spyctral.fourier as fs
import numpy as np
from numpy import exp, sin, cos, dot
from numpy.linalg import norm
import scipy as sp
from scipy import pi,rand,randn

N = 101
gamma = int(10*rand())
delta = int(10*rand())
gd = gamma+delta + int(N%2==0)
L = 10*rand()
a = 10*randn()
f = lambda x: exp(sin(pi*x))

[t,w] = fs.quad.gq(N,scale=L,shift=a)
ks = spyctral.common.indexing.integer_range(N)
funs = fs.eval.fseries(t,ks,gamma=gamma,delta=delta,scale=L,shift=a)
weight = fs.weights.weight(t,gamma=gamma,delta=delta,scale=L,shift=a)

fft_overhead = fs.fft.fft_overhead(N,gamma=gamma,delta=delta,scale=L)

modes = dot(funs.T.conj()*w*weight,f(t))

modes_fft = fs.fft.fft(f(t),gamma=gamma,delta=delta,scale=L)
modes_fft_onoff = fs.fft.fft_online(f(t),fft_overhead)

print "Modal fft error is", norm(modes_fft[gd:-gd]-modes[gd:-gd])
print "Modal fft on/off error is", norm(modes_fft_onoff[gd:-gd]-modes[gd:-gd])

rec_fft = fs.fft.ifft(modes_fft,gamma=gamma,delta=delta,scale=L)
rec_fft_onoff = fs.fft.ifft_online(modes_fft,fft_overhead)

print "Reconstruction error is", norm(f(t) - rec_fft)
print "Reconstruction error on/off is", norm(f(t) - rec_fft_onoff)

alpha = -1/2.; beta = -1/2.; A = delta; B = gamma;
from spyctral.jacobi.jfft import rmatrix_entries, rmatrix_invert, rmatrix_entries_apply, rmatrix_entries_invert, rmatrix_apply_seq
from spyctral.fourier.connection import int_connection, int_connection_backward
u = rand(N)
from spyctral.jacobi.jfft import rmatrix_invert as jconnection_inv
from spyctral.jacobi.jfft import rmatrix_apply_seq as jconnection
from spyctral.fourier.connection import sc_collapse, sc_expand
from numpy import append
from scipy import conj
G = gamma; D = delta; delta = 0; gamma=0;

NEven = (N%2)==0
ucopy = u.copy()
if NEven:
    N += 1
    ucopy[0] /= 2.
    ucopy = append(ucopy,conj(ucopy[0]))

[cmodes,smodes] = sc_collapse(ucopy,N)
cmodes = jconnection(cmodes,delta-1/2.,gamma-1/2.,D,G)
smodes = jconnection(smodes,delta+1/2.,gamma+1/2.,D,G)

utemp = sc_expand(cmodes,smodes,N)
if NEven:
    utemp[0] += conj(utemp[-1])
    utemp = utemp[:-1]

cmodes3 = jconnection_inv(cmodes,delta-1/2.,gamma-1/2.,D,G)
smodes3 = jconnection_inv(smodes,delta+1/2.,gamma+1/2.,D,G)

u2 = sc_expand(cmodes3,smodes3,N)

utemp2 = int_connection(u,gamma=0.,delta=0.,G=G,D=D)
u3 = int_connection_backward(utemp2,gamma=0.,delta=0.,G=G,D=D)
