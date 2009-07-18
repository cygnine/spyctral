from spyctral.classify import *
import spyctral.mapjpoly as mapjpoly

class MappedJacobiPolynomialQuadrature(QuadratureRule):

    def __init__(self,N=0,s=1.,t=1.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Mapped Jacobi polynomial"
        self.parameters = {"s":s, "t":t, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = mapjpoly.quad.gq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        return mapjpoly.weights.weight(x,**self.parameters)

class MappedJacobiFunctionQuadrature(QuadratureRule):

    def __init__(self,N=0,s=1.,t=1.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Mapped Jacobi function"
        self.parameters = {"s":s, "t":t, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = mapjpoly.quad.pgq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        from numpy import ones
        return ones(x.size)

MappedJacobiQuadrature = MappedJacobiFunctionQuadrature

class MappedJacobiPolynomialBasis(WholeSpectralBasis):
    """ Mapped Jacobi polynomial basis information """

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1., t=1., shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.basis_type = "Mapped Jacobi polynomial"
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return MappedJacobiPolynomialQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return mapjpoly.eval.jacobi_function(x,n,**self.parameters)

    def derivative(self,x,n):
        return mapjpoly.eval.djacobi_function(x,n,**self.parameters)

class MappedJacobiFunctionBasis(WholeSpectralBasis):
    """ Mapped Jacobi function expansion basis. """

    basis_type = "Mapped Jacobi function"

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,s=1., t=1., shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.N = N
        self.parameters = {'s':s, 't':t, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def fft_overhead(self):
        return mapjpoly.fft.mjwfft_overhead(selfN,**self.parameters)
    def fft_online(self,fx):
        return mapjpoly.fft.mjwfft_online(fx,self.fft_overhead_data)

    def canonical_quadrature(self):
        return MappedJacobiFunctionQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return mapjpoly.eval.weighted_jacobi_function(x,n,**self.parameters)

    def derivative(self,x,n):
        return mapjpoly.eval.dweighted_jacobi_function(x,n,**self.parameters)

#    def gauss_quadrature(self):
#        return hermite.quad.pgq(self.N,**self.parameters)

MappedJacobiBasis = MappedJacobiFunctionBasis
