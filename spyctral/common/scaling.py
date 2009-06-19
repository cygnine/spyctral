def scale_factor(L,x,delta=0.5):
    """
    Returns the scaling factor necessary so that at least [delta] percent of the
    Gaussian nodes [x] satisfy | [x] |<= [L]. Mostly used for the infinite
    interval expansions.
    """

    from numpy import floor, ceil,abs,mean

    N = x.size
    delta = min(1.0,delta)
    
    Nfrac = ceil(N*delta)

    # Find scale interval [Lprev,Lnext] containing desired delta
    Lprev = 1e6
    Lnext = Lprev
    Nxs = sum(abs(x*Lnext)<=L)
    while Nxs<Nfrac:
        Lprev = Lnext
        Lnext /= 2
        Nxs = sum(abs(x*Lnext)<=L)

    # Find point in [Lprev,Lnext] at which number equals/exceeds Nfrac:
    Lmiddle = mean([Lprev,Lnext])
    Lsep = Lprev-Lmiddle
    Ltol = 1e-3
    while Lsep>Ltol:
        Nxs = sum(abs(x*Lmiddle)<=L)
        if Nxs<Nfrac:
            Lprev = Lmiddle
        else:
            Lnext = Lmiddle
        Lmiddle = mean([Lprev,Lnext])
        Lsep = Lprev-Lmiddle

    # Return a safe amount
    return Lnext
