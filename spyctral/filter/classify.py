import spyctral.filter

class Filter:

    got_modal_weights = False

    def __init__(N=0.):
        from numpy import zeros
        self.etas = zeros(N)
        self.modal_weights = zeros(N)

    def get_modal_weights(self,etas):
        return etas;

    def initialize_modal_weights(self):
        self.modal_weights = self.get_modal_weights(self.etas)
        self.got_modal_weights = True

    def apply_filter(self,input):
        if not self.got_modal_weights:
            self.modal_weights = self.get_modal_weights(self.etas)
            self.got_modal_weights = True
        return self.modal_weights*input

class ExponentialFilter(Filter):
    import numpy
    from numpy import log

    def __init__(self,truncation_value = -log(numpy.finfo(float).eps),
            filter_order=8., preservation_ratio=0.5):

        self.parameters = {'alpha': truncation_value,
                'p': filter_order,
                'eta_cutoff': preservation_ratio}

    def get_modal_weights(self,etas):
        from spyctral.filter import exponential
        return exponential.modal_weights(etas,**self.parameters)

class CesaroFilter(Filter):
    def __init__(self):
        pass

    def get_modal_weights(self,etas):
        from spyctral.filter import cesaro
        return cesaro.modal_weights(etas)

class LanczosFilter(Filter):

    def __init__(self):
        pass

    def get_modal_weights(self,etas):
        from spyctral.filter import lanczos
        return lanczos.modal_weights(etas)

class RaisedCosineFilter(Filter):
    def __init__(self,filter_order=1.):
        self.parameters = {'order':filter_order}

    def get_modal_weights(self,etas):
        from spyctral.filter import raised_cosine
        return raised_cosine.modal_weights(etas,**self.parameters)
