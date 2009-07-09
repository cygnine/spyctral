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

class HermitePolynomialBasis(SpectralBasis):
    """ Hermite polynomial basis information """

    def __init__(self,N=0,quadrature=None,interpolation_nodes=None,
                 filter=None,mu=0.,shift=0.,scale=1.):
        self.N = N
        self.mu = mu;
        self.basis_type = "Hermite polynomial"
        self.indexing_type = "whole"
        self.default_qudrature
        self.assign_indices()

        self.parameters = {'mu':mu, 'scale':scale,
                'shift':shift}

        if interpolation_nodes is None:
            if quadrature is not None:
                self.quadrature = quadrature
                self.nodes = self.quadrature.nodes
            else:
                self.quadrature = HermitePolynomialQuadrature(N=self.N,\
                                    **self.parameters)
                self.nodes = self.quadrature.nodes
        else:
            self.nodes = interpolation_nodes
            self.make_vandermonde()
            self.make_differentiation_matrix()

    def evaluation(self,x,n):
        return hermite.eval.hermite_polynomial(x,n,**self.parameters)

    def derivative(self,x,n):
        return hermite.eval.hermite_polynomial(x,n,d=1,**self.parameters)

    def gauss_quadrature(self):
        return hermite.quad.gq(self.N,**self.parameters)
