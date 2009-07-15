from spyctral.classify import *
import spyctral.wiener as wiener

class WeightedWienerQuadrature(QuadratureRule):

    def __init__(self,N=0,s=1.,t=0.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Weighted Wiener"
        self.parameters = {"s":s, "t":t, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = wiener.quad.pgq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        from numpy import ones
        return ones(x.size)

class UnweightedWienerQuadrature(QuadratureRule):

    def __init__(self,N=0,s=1.,t=0.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Unweighted Wiener"
        self.parameters = {"s":s, "t":t, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = wiener.quad.pgq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        from numpy import ones
        return wiener.weights.weight(x,**self.parameters)

WienerQuadrature = WeightedWienerQuadrature

class UnweightedWienerBasis(IntegerSpectralBasis):
    """ Unweighted Wiener rational functions basis. """

    basis_type = "Unweighted Wiener rational function"

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1.,t=0.,shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return UnweightedWienerQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return wiener.eval.wiener(x,n,**self.parameters)

    def derivative(self,x,n):
        return wiener.eval.dwiener(x,n,**self.parameters)

class WeightedWienerBasis(FFTBasis,IntegerSpectralBasis):
    """ Weighted Wiener rational functions basis. """

    basis_type = "Weighted Wiener rational function"

    def fft_overhead(self):
        return wiener.fft.fft_collocation_overhead(self.N,**self.parameters)
    def fft_online(self,fx):
        return wiener.fft.fft_collocation_online(fx,self.fft_overhead_data)
    def ifft_online(self,fx):
        return wiener.fft.ifft_collocation_online(fx,self.fft_overhead_data)
    def fft_nodal_set(self):
        temp = WeightedWienerQuadrature(N=self.N,s=1,t=self.parameters['t'],
                 scale=self.parameters['scale'], shift=self.parameters['shift'])
        return temp.nodes

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1.,t=0.,shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return WeightedWienerQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return wiener.eval.weighted_wiener(x,n,**self.parameters)

    def derivative(self,x,n):
        return wiener.eval.dweighted_wiener(x,n,**self.parameters)

    def make_stiffness_matrix(self):
        self.stiffness_matrix = wiener.matrices.weighted_wiener_stiffness_matrix(self.N,
                **self.parameters)

WienerBasis = WeightedWienerBasis
