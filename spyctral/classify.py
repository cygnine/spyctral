class SpectralBasis:
    """ The basic class type for all spectral basis expansions. """

    def __init__(self,N=0):
        self.N = N
        self.basis_type = None
        self.affine_shift = 0.
        self.affine_scale = 1.
        self.quadrature = None
        indexing_type = None;
        indexing_function = None;

    def __str__(self):
        return str(self.basis_type) + \
                " Spectral basis expansion with %d degrees of freedom" % self.N

    def __repr__(self):
        return str(self.basis_type) + \
                " Spectral basis expansion with %d degrees of freedom" % self.N

    def assign_indices(self):
        if self.indexing_type.lower() == "whole":
            self.indexing_function = spyctral.common.indexing.whole_range
        elif self.indexing_type.lower() == "integer":
            self.indexing_function = spyctral.common.indexing.integer_range
        else:
            raise ValueError("Error: I did not recognize the indexing type %s"\
                    % self.indexing_type)

        self.indices = self.indexing_function(self.N)

    def make_vandermonde(self):
        self.vandermonde = self.evaluation(self.nodes,self.indices)

    def make_vandermonde_inverse(self):
        from numpy.linalg import inv
        if self.quadrature is None:
            self.vandermonde_inverse = inv(self.vandermonde)
        else:
            self.vandermonde_inverse = \
                    (self.vandermonde.T.conj()*self.quadrature.weights);

    def make_differentiation_matrix(self):
        from numpy import dot
        self.differentiation_matrix = dot(self.derivative(self.nodes,self.indices), \
                 self.vandermonde_inverse)

class QuadratureRule:
    """ The basic class type for all quadrature rule instantiations. """

    def __init__(self,N=0):
        self.N = N
        self.nodes = None
        self.weights = None
        self.weight_function = None
        self.quadrature_type = None

    def __str__(self):
        return str(self.quadrature_type) + \
          " quadrature rule with %d degrees of freedom" % self.N

    def __repr__(self):
        return str(self.quadrature_type) + \
          " quadrature rule with %d degrees of freedom" % self.N
