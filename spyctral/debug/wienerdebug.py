from spyctral import wienerfun as wf
from numpy import dot, exp
from numpy.linalg import norm
import pylab as pl
import pudb

Nq = 200
N = 200
L = 10.
ks = wf.quad.N_to_ks(N)
[x,w] = wf.quad.genwienerw_pgquad(Nq,scale=L)

wfs = wf.eval.genwienerw(x,ks,scale=L)

mass = dot(wfs.T.conj()*w,wfs)

f = lambda x: exp(-(x-1)**2)
df = lambda x: -2*(x-1)*exp(-(x-1)**2)
fx = f(x)
dfx = df(x)

modes = dot(wfs.T.conj()*w,fx)

Ne = 1000
[xe,we] = wf.quad.genwienerw_pgquad(Ne,scale=L)
wfse = wf.eval.genwienerw(xe, ks,scale=L)
dwfse = wf.eval.dgenwienerw(xe, ks,scale=L)
stiff = wf.coeffs.genwienerw_stiff(N,scale=L)

fft_overhead = wf.fft.fft_nodes_to_modes_overhead(Nq,scale=L)

rec= dot(wfse,modes)
rec2 = dot(wfse,wf.fft.fft_nodes_to_modes(fx,scale=L))
rec3 = dot(wfse,wf.fft.fft_nodes_to_modes_online(fx,fft_overhead))
rec4 = wf.fft.fft_modes_to_nodes(wf.fft.fft_nodes_to_modes(fx,scale=L),scale=L)
rec5 = wf.fft.fft_modes_to_nodes_online(wf.fft.fft_nodes_to_modes_online(fx,fft_overhead),fft_overhead)
drec = dot(dwfse,modes)
drec2 = dot(wfse,stiff*modes)

print "Reconstruction error", norm(rec-f(xe))
print "Derivative error", norm(drec-df(xe))
print "Stiff error", norm(drec2-df(xe))
print "FFT error", norm(rec2-f(xe))
print "FFT off/on error", norm(rec3-f(xe))
print "FFT2 error", norm(rec4-f(x))
print "FFT2 on/off error", norm(rec5-f(x))
