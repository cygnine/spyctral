# Filter package for spectral methods

__all__ = []

import exponential
import lanczos
import raised_cosine
import cesaro

#__all__ += exp.__all__

def filter_parse(filter_type):
    if filter_type == 'exponential':
        return exponential
    elif filter_type == 'lanczos':
        return lanczos
    elif filter_type == 'raised_cosine':
        return raised_cosine
    elif filter_type == 'cesaro':
        return cesaro
    else:
        return None
