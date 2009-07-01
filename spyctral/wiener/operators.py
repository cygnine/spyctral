
def fft_nodal_differentiation(fx,**kwargs):
    """Given the nodal function values, returns the nodal approximations to the
    derivative computed using:
        1.) the fft
        2.) application of the sparse stiffness matrix to the nodes
        3.) the ifft
    s must be an integer for all this to work.
    """

    from spyctral.wiener.matrices import weighted_wiener_stiffness_matrix \
            as stiffmat
    from spyctral.wiener.fft import fft_collocation as fft
    from spyctral.wiener.fft import ifft_collocation as ifft

    stiff = stiffmat(len(fx),**kwargs)
    
    if fx.dtype==object:
        from pymbolic.algorithm import csr_matrix_multiply
        temp = fft(fx,**kwargs)
        temp = csr_matrix_multiply(stiff,temp)
        return ifft(temp,**kwargs)
    else:
        return ifft(stiff*fft(fx,**kwargs),**kwargs)
