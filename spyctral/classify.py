import spyctral

class SpectralBasis:
    """ The basic class type for all spectral basis expansions. """

    basis_type = None
    quadrature = None
    vandermonde = None
    vandermonde_inverse = None
    nodal_differentiation_matrix = None
    parameters = {}
    indexing_type = None;
    indexing_function = None;
    fftable = False
    fft_initialized= False
    default_quadrature = lambda N: None

    def __init__(self,N=0):
        self.N = N


    def __str__(self):
        return str(self.basis_type) + \
                " spectral basis expansion with %d degrees of freedom" % self.N

    def __repr__(self):
        return str(self.basis_type) + \
                " spectral basis expansion with %d degrees of freedom" % self.N

    def initialize_quadrature(self,interpolation_nodes,quadrature):
        if interpolation_nodes is None:
            if quadrature is not None:
                self.quadrature = quadrature
            else:
                self.quadrature = self.default_quadrature(self.N,
                                    **self.parameters)
            self.nodes = self.quadrature.nodes
        else:
            self.nodes = interpolation_nodes
            # Creates vandermonde and vandermonde_inv by default
            self.make_nodal_differentiation_matrix()

    def make_vandermonde(self):
        """
        Utilizes the evaluation routine self.evaluation along with the nodes
        specified in self.nodes and the indices self.indices to create the
        Vandermonde matrix, stored in self.vandermonde.
        """
        self.vandermonde = self.evaluation(self.nodes,self.indices)

    def initialize_fft(self):
        """
        If the basis set supports the FFT, this function performs and stores any
        overhead required to perform the FFT.
        """
        if self.fftable:
            self.__fft_overhead = self.fft_overhead(self.N,**self.parameters)
            self.fft_initialized = True
        else:
            print "This basis set does not support the FFT\n"

    def fft(self,x):
        """
        Takes as input nodal evaluations at the FFT grid points and produced
        modal coefficients corresponding to the basis expansion.
        """
        if self.fftable:
            if not self.fft_initialized:
                self.initialize_fft()
            return self.fft_online(x,**self.parameters)
        else:
            print "This basis set does not support the FFT\n"
            return None

    def ifft(self,x):
        """
        Takes as input modal coefficients corresponding to the basis and outputs
        nodal evaluations at the FFT grid points.
        """
        if self.fftable:
            if not self.fft_initialized:
                self.initialize_fft()
            return self.ifft_online(x,**self.parameters)
        else:
            print "This basis set does not support the FFT\n"
            return None

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

    def make_nodal_differentiation_matrix(self):
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

class WholeSpectralBasis(SpectralBasis):
    """ 
    SpectralBasis instance for which the modes are indexed as whole numbers
    """

    def assign_indices(self):
        """
        Assigns default whole-number indices derived from self.N
        """
        self.indexing_function = spyctral.common.indexing.whole_range
        self.indices = self.indexing_function(self.N)
        self.modal_fractions = self.indices/float(self.N-1)

class IntegerSpectralBasis(SpectralBasis):
    """ 
    SpectralBasis instance for which the modes are indexed as integers
    """

    def assign_indices(self):
        """
        Assigns default integer-number indices derived from self.N
        """
        self.indexing_function = spyctral.common.indexing.integer_range
        self.indices = self.indexing_function(self.N)
        self.modal_fractions = self.indices/float(int(self.N)/2)

class QuadratureRule:
    """ The basic class type for all quadrature rule instantiations. """

    def __init__(self,N=0):
        self.N = N

    nodes = None
    weights = None
    weight_function = None
    quadrature_type = None

    def __str__(self):
        return str(self.quadrature_type) + \
          " quadrature rule with %d degrees of freedom" % self.N

    def __repr__(self):
        return str(self.quadrature_type) + \
          " quadrature rule with %d degrees of freedom" % self.N
