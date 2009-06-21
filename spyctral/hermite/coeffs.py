# Returns recurrence coefficients for certain values of n
def recurrence(ns,mu=0):

    from numpy import arange, array, zeros
    from scipy.special import gamma

    ns = array(ns)
    N = ns.size
    ns = ns.reshape([N])

    a_s = zeros([N])
    b_s = zeros([N])

    b_s[ns==0] = gamma(mu+1/2.)
    b_s[ns>=1] = 1/2.*(arange(1,N))
    ks = range(1,N,2)
    b_s[ks] += mu
    
    return [a_s,b_s]

def recurrence_range(N,mu=0.):
    return recurrence(range(N),mu=mu)
