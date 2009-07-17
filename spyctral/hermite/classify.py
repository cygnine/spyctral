from spyctral.classify import *
import spyctral.hermite as hermite

class HermitePolynomialQuadrature(QuadratureRule):

    def __init__(self,N=0,mu=0.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Hermite polynomial"
        self.parameters = {"mu":mu, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = hermite.quad.gq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        return hermite.weights.weight(x,**self.parameters)

class HermiteFunctionQuadrature(QuadratureRule):

    def __init__(self,N=0,mu=0.,shift=0.,scale=1.):
        self.N = N
        self.quadrature_type = "Hermite function"
        self.parameters = {"mu":mu, "scale":scale, "shift":shift}
        if N>0:
            [self.nodes,self.weights] = hermite.quad.pgq(N,**self.parameters)

    def weight_function(self,x):
        """
        Evaluates the weight function associated with the quadrature rule.
        """
        from numpy import ones
        return ones(x.size)

HermiteQuadrature = HermiteFunctionQuadrature

class HermitePolynomialBasis(WholeSpectralBasis):
    """ Hermite polynomial basis information """

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,mu=0.,shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.basis_type = "Hermite polynomial"
        self.N = N
        self.parameters = {'mu':mu, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return HermitePolynomialQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return hermite.eval.hermite_polynomial(x,n,**self.parameters)

    def derivative(self,x,n):
        return hermite.eval.hermite_polynomial(x,n,d=1,**self.parameters)

#    def gauss_quadrature(self):
#        return hermite.quad.gq(self.N,**self.parameters)

class HermiteFunctionBasis(WholeSpectralBasis):
    """ Hermite function expansion basis. """

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,mu=0.,shift=0.,scale=1.,
                 physical_scale=None,physical_scale_ratio=0.8):
        self.basis_type = "Hermite function"
        self.N = N
        self.parameters = {'mu':mu, 'scale':scale,
                'shift':shift}
        self.assign_indices()
        if physical_scale is not None:
            self.scale_nodes(physical_scale,physical_scale_ratio)
        self.initialize_quadrature(interpolation_nodes,quadrature)
        self.make_nodal_differentiation_matrix()

    def canonical_quadrature(self):
        return HermiteFunctionQuadrature(N=self.N,**self.parameters)

    def evaluation(self,x,n):
        return hermite.eval.hermite_function(x,n,**self.parameters)

    def derivative(self,x,n):
        return hermite.eval.dhermite_function(x,n,**self.parameters)

    def make_stiffness_matrix(self):
        self.stiffness_matrix = hermite.matrices.stiffness_matrix(self.N,
                **self.parameters)

#    def gauss_quadrature(self):
#        return hermite.quad.pgq(self.N,**self.parameters)

HermiteBasis = HermiteFunctionBasis
