# Generates the NxN stiffness matrix for weighted Wiener functions (for N even,
# the indices are negatively-biased)
# Returns a sparse CSR matrix
def weighted_wiener_stiffness_matrix(N,s=1.,t=0.,scale=1.):
    from spyctral.wiener.coeffs import weighted_wiener_stiff_entries
    from numpy import zeros, array, sign, abs
    from scipy import sparse

    def sgn(x):
        return (x==0)+sign(x)

    N = int(N)
    if bool(N%2):
        k = (N-1)/2
        ks = range(-k,k+1)
        kbottom = -k
    else:
        k = N/2
        ks = range(-k,k)
        kbottom = -k

    entries = weighted_wiener_stiff_entries(ks,s,t,scale=scale)
    I = []
    J = []
    V = []

    stiff = zeros([N,N])
    kcount = 0
    for k in ks:
        kvee = sgn(k)*(abs(k)-1)
        kwedge = sgn(k)*(abs(k)+1)
        kinds = list(array([kvee,-kvee,k,-k,kwedge,-kwedge])-kbottom)
        thisk = k-kbottom
        lcount = 0
        rowks = []
        colks = []
        es = []

        for l in kinds:
            if 0<=l<=N-1:
                rowks.append(thisk)
                es.append(entries[k-kbottom,lcount])
                colks.append(kinds[lcount])
            #else:
                #colks.pop(lcount)
                #kinds.pop(lcount)
            lcount += 1

        map(I.append,rowks)
        map(J.append,colks)
        map(V.append,list(es))


    I = array(I)
    J = array(J)
    V = array(V)
    # Actually, since defintion of stiffness matrix is <phi,dphi>, the I's are
    # column indices, and the J's are row indices
    return sparse.coo_matrix((V,(J,I)),shape=(N,N)).tocsr()
    #return [rowks,colks,es]
