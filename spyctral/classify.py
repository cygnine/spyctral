import spyctral

class SpectralBasis:
    """ The basic class type for all spectral basis expansions. """

    def __init__(self,N=0):
        self.N = N
        self.basis_type = None
        self.shift = 0.
        self.scale = 1.
        self.quadrature = None
        self.vandermonde = None
        self.vandermonde_inverse = None
        self.nodal_differentiation_matrix = None
        self.parameters = {}
        self.indexing_type = None;
        self.indexing_function = None;
        self.default_quadrature = lambda N: None

    def __str__(self):
        return str(self.basis_type) + \
                " Spectral basis expansion with %d degrees of freedom" % self.N

    def __repr__(self):
        return str(self.basis_type) + \
                " Spectral basis expansion with %d degrees of freedom" % self.N

    def assign_indices(self):
        """
        Assigns default indices derived from self.N and self.indexing_type.
        """
        if self.indexing_type.lower() == "whole":
            self.indexing_function = spyctral.common.indexing.whole_range
        elif self.indexing_type.lower() == "integer":
            self.indexing_function = spyctral.common.indexing.integer_range
        else:
            raise ValueError("Error: I did not recognize the indexing type %s"\
                    % self.indexing_type)

        self.indices = self.indexing_function(self.N)

    def make_vandermonde(self):
        """
        Utilizes the evaluation routine self.evaluation along with the nodes
        specified in self.nodes and the indices self.indices to create the
        Vandermonde matrix, stored in self.vandermonde.
        """
        self.vandermonde = self.evaluation(self.nodes,self.indices)

    def make_vandermonde_inverse(self):
        """
        Constructs the inverse of the Vandermonde matrix.
        If there is no quadrature rule specified, this function uses
        numpy.linalg.inv to compute the inverse of self.vandermonde. If a
        quadrature rule is specified, it assumes that the rule is accurate
        enough that the mass matrix is produced. With this assumption, the
        inverse of the vandermonde matrix can be computed by simply applying the
        quadrature rule to the Vandermonde matrix.
        """
        from numpy.linalg import inv
        if self.vandermonde is None:
            self.make_vandermonde()
        if self.quadrature is None:
            self.vandermonde_inverse = inv(self.vandermonde)
        else:
            self.vandermonde_inverse = \
                    (self.vandermonde.T.conj()*self.quadrature.weights);

    def make_differentiation_matrix(self):
        """ 
        Uses the methods self.derivative, self.nodes, and
        self.vandermonde_inverse to construct the nodal differentiation matrix.
        """
        from numpy import dot
        if self.vandermonde is None:
            self.make_vandermonde()
        if self.vandermonde_inverse is None:
            self.make_vandermonde_inverse()
        self.nodal_differentiation_matrix = dot(self.derivative(self.nodes,self.indices), \
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
