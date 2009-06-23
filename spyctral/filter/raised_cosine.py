def modal_weights(etas,order=1):
    """Returns the modal coefficient attentuation factors for the raised cosine
    spectral filter"""

    from numpy import cos
    from scipy import pi

    coeffs = (1/2.)*(1+cos(pi*etas))

    return coeffs**order
