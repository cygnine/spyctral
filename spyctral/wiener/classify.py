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

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1.,t=0.,shift=0.,scale=1.,
                 physical_scale=None,delta=0.8):
        self.basis_type = "Unweighted Wiener rational function"
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,delta)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return UnweightedWienerQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return wiener.eval.wiener(x,n,**self.parameters)

    def derivative(self,x,n):
        return wiener.eval.dwiener(x,n,**self.parameters)

class WeightedWienerBasis(IntegerSpectralBasis):
    """ Weighted Wiener rational functions basis. """

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1.,t=0.,shift=0.,scale=1.,
                 physical_scale=None,delta=0.8):
        self.basis_type = "Weighted Wiener rational function"
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,delta)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return WeightedWienerQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return wiener.eval.weighted_wiener(x,n,**self.parameters)

    def derivative(self,x,n):
        return wiener.eval.dweighted_wiener(x,n,**self.parameters)

WienerBasis = WeightedWienerBasis
