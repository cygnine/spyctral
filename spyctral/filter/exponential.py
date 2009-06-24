# Exponential filter module for spectral filter package

__all__ = []
from numpy import log
#from eps import eps

# Returns attenuation coefficients for exponential coefficients
def modal_weights(etas,alpha=-log(1e-10),p=8.,eta_cutoff=0.5):
    
    from numpy import abs,exp,ones

    # Flags for preservation
    etas = abs(etas)
    pflags = etas<=eta_cutoff

    coeffs = ones(etas.size)

    factor = -alpha*( (etas[~pflags]-eta_cutoff)/(1-eta_cutoff) )**p
    coeffs[~pflags] = exp(factor)
    return coeffs
