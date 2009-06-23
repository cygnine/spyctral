# Exponential filter module for spectral filter package

__all__ = []
from numpy import log
from eps import eps

# Returns attenuation coefficients for exponential coefficients
def modal_weights(etas,alpha=-log(eps),p=8.,etac=0.5):
    
    from numpy import abs,exp,ones

    # Flags for preservation
    etas = abs(etas)
    pflags = etas<=etac

    coeffs = ones(etas.size)

    factor = -alpha*( (etas[~pflags]-etac)/(1-etac) )**p
    coeffs[~pflags] = exp(factor)
    return coeffs