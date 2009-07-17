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
    default_quadrature = lambda N: None
    canonical_quadrature = None

    stiffness_matrix = None
    mass_matrix = None

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
                self.quadrature = self.canonical_quadrature()
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


    def rehash_parameters(self,**kwargs):
        """ 
        Rehashes self.parameters to reflect changes made to input parameters.
        """
        for key in kwargs.keys():
            self.parameters[key] = kwargs[key]

        self.quadrature = None
        self.vandermonde = None
        self.vandermonde_inverse = None
        self.nodal_differentiation_matrix = None
        self.stiffness_matrix = None

    def scale_nodes(self,L,physical_scale_ratio):
        """
        Sets the affine scaling factor self.scale so that (physical_scale_ratio x N) of the
        canonical nodes lie inside [-L,L]. 
        """
        from spyctral.common.scaling import scale_factor

        if self.canonical_quadrature is None:
            print "Error: cannot scale...I don't have a canonical set of nodes"
        else:
            x = self.canonical_quadrature().nodes
            scale = scale_factor(L,x,delta=physical_scale_ratio)
            self.rehash_parameters(scale=scale)
            self.initialize_quadrature(None,None)
            self.make_nodal_differentiation_matrix()

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

    def initialize_filter(self,filter=None):
        if filter is None:
            # Default filter type:
            print "Using default Exponential filter..."
            self.filter = spyctral.ExponentialFilter()
        elif isinstance(filter,spyctral.Filter):
            self.filter = filter
        else:
            raise TypeError("You must initialize a filter with a Filter \
                class-type, or with None for the default")
            return None
        self.modal_fractions = self.filter_fractions(self.N)
        self.filter.initialize_modal_weights(self.modal_fractions)

    def apply_stiffness_matrix(self,modes):
        from scipy.sparse.csr import csr_matrix
        from numpy import ndarray, dot
        if self.stiffness_matrix is None:
            self.make_stiffness_matrix()
        if isinstance(self.stiffness_matrix,csr_matrix):
            try: 
                from pymbolic.algorithm import csr_matrix_multiply
                return csr_matrix_multiply(self.stiffness_matrix,modes)
            except:
                return self.stiffness_matrix*modes
        elif isinstance(self.stiffness_matrix,ndarray):
            return dot(self.stiffness_matrix,modes)
        else:
            raise TypeError("Unrecognized type %s for stiffness matrix" % \
                    type(self.stiffness_matrix))

    def apply_spectral_filter_to_modes(self,modes):
        return self.filter.apply_filter(modes)

    def make_spectral_filter_matrix_for_nodes(self):
        from numpy import dot, diag
        self.spectral_filter_matrix_for_nodes = dot(self.vandermonde,
                dot(diag(self.filter.modal_weights), self.vandermonde_inverse))

    def apply_spectral_filter_to_nodes(self,fx):
        """ Use this if you don't want to explicitly save the filter matrix
        operator for some reason."""
        from numpy import dot, diag
        temp = dot(self.vandermonde, dot(diag(self.filter.modal_weights),
            self.vandermonde_inverse))
        return dot(temp,fx)

class FFTBasis:
    """
    Spectral basis method that can use the FFT for modal-nodal transformations.
    """

    fftable = True
    fft_initialized= False

    def initialize_fft(self):
        """
        If the basis set supports the FFT, this function performs and stores any
        overhead required to perform the FFT.
        """
        if self.fftable:
            self.rehash_parameters()
            self.initialize_quadrature(None,self.fft_quadrature_rule())
            self.fft_overhead_data = self.fft_overhead()
            self.fft_initialized = True
        else:
            print "This nodal set does not support the FFT\n"

    def fft(self,fx):
        """
        Takes as input nodal evaluations at the FFT grid points and produced
        modal coefficients corresponding to the basis expansion.
        """
        if self.fftable:
            if not self.fft_initialized:
                self.initialize_fft()
            return self.fft_online(fx)
        else:
            print "This basis set does not support the FFT\n"
            return None

    def ifft(self,F):
        """
        Takes as input modal coefficients corresponding to the basis and outputs
        nodal evaluations at the FFT grid points.
        """
        if self.fftable:
            if not self.fft_initialized:
                self.initialize_fft()
            return self.ifft_online(F)
        else:
            print "This basis set does not support the FFT\n"
            return None

    def fft_differentiation(self,fx):
        """
        Uses the FFT to turn nodal evaluations fx into modal coefficients,
        applies the stiffness matrix, and then uses to the IFFT to take things
        back to nodal evaluations.
        """
        if not self.fft_initialized:
            print "You must first initialize the FFT"
        else:
            return self.ifft(self.apply_stiffness_matrix(self.fft(fx)))

    def apply_spectral_filter_to_nodes(self,fx):
        """ Use this if you don't want to explicitly build/save the filter matrix
        operator."""
        
        if self.fft_initialized:
            return self.ifft(self.apply_spectral_filter_to_modes(self.fft(fx)))
        else:
            from numpy import dot, diag
            temp = dot(self.vandermonde, dot(diag(self.filter.modal_weights),
                self.vandermonde_inverse))
            return dot(temp,fx)

class WholeSpectralBasis(SpectralBasis):
    """ 
    SpectralBasis instance for which the modes are indexed as whole numbers
    """

    def assign_indices(self):
        """
        Assigns default whole-number indices derived from self.N
        """
        self.indexing_function = spyctral.common.indexing.whole_range
        self.filter_fractions = spyctral.common.indexing.whole_etas
        self.indices = self.indexing_function(self.N)
        #self.modal_fractions = self.indices/float(self.N-1)



class IntegerSpectralBasis(SpectralBasis):
    """ 
    SpectralBasis instance for which the modes are indexed as integers
    """

    def assign_indices(self):
        """
        Assigns default integer-number indices derived from self.N
        """
        self.indexing_function = spyctral.common.indexing.integer_range
        self.filter_fractions = spyctral.common.indexing.integer_etas
        self.indices = self.indexing_function(self.N)
        #self.modal_fractions = self.indices/float(int(self.N)/2)

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
