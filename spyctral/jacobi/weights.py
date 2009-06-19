
def weight(r,alpha=-1/2.,beta=-1/2.,shift=0.,scale=1.):
    from spyctral.common.maps import standard_scaleshift as sss
    from spyctral.common.maps import physical_scaleshift as pss
    sss(r,shift=shift,scale=scale)
    wfun = (1-r)**alpha*(1+r)**beta
    pss(r,shift=shift,scale=scale)

    return wfun
