import spyctral.filter

class Filter:

    got_modal_weights = False

    def __init__(N=0.):
        from numpy import zeros
        self.modal_fractions = zeros(N)
        self.modal_weights = zeros(N)

    def get_modal_weights(self,modal_fractions):
        return modal_fractions;

    def initialize_modal_weights(self,modal_fractions):
        self.modal_weights = self.get_modal_weights(modal_fractions)
        self.got_modal_weights = True

    def apply_filter(self,input):
        if not self.got_modal_weights:
            raise RuntimeError('You must initialize the filter first')
        return self.modal_weights*input

class ExponentialFilter(Filter):
    import numpy

    def __init__(self,N=0,truncation_value = numpy.finfo(float).eps,
            filter_order=8., preservation_ratio=0.5):
        from numpy import log

        self.parameters = {'alpha': -log(truncation_value),
                'p': filter_order,
                'eta_cutoff': preservation_ratio}

    def get_modal_weights(self,modal_fractions):
        from spyctral.filter import exponential
        return exponential.modal_weights(modal_fractions,**self.parameters)

class CesaroFilter(Filter):
    def __init__(self):
        pass

    def get_modal_weights(self,modal_fractions):
        from spyctral.filter import cesaro
        return cesaro.modal_weights(modal_fractions)

class LanczosFilter(Filter):

    def __init__(self):
        pass

    def get_modal_weights(self,modal_fractions):
        from spyctral.filter import lanczos
        return lanczos.modal_weights(modal_fractions)

class RaisedCosineFilter(Filter):
    def __init__(self,filter_order=1.):
        self.parameters = {'order':filter_order}

    def get_modal_weights(self,modal_fractions):
        from spyctral.filter import raised_cosine
        return raised_cosine.modal_weights(modal_fractions,**self.parameters)
