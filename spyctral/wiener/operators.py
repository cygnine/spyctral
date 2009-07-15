
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

    return ifft(stiff*fft(fx,**kwargs),**kwargs)

def apply_stiffness_matrix(modes,**kwargs):
    """
    Generates and applies the stiffness matrix to the modal coefficients.
    """
    from spyctral.wiener.matrices import weighted_wiener_stiffness_matrix as \
        stiffmat
    stiff = stiffmat(len(modes),**kwargs)

    try:
        from pymbolic.algorithm import csr_matrix_multiply as csr_mult
        return csr_mult(stiff,modes)
    except:
        return stiff*mdoes
