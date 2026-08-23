
# Python distribution modules
import numpy as np

# T10 : specialized float32 Simplex compute kernel for the GMN default
# generative path. Reproduces pyEDM 2.5.7 Simplex numerics exactly to
# 1e-5 ( parity contract ) with no pyEDM / pandas / kd-tree per step :
#
#   - brute-force Euclidean knn against the frozen NodeLibrary ( one
#     query point per step ; a kd-tree never amortizes its build here )
#   - deterministic tie ordering : distance, then |predRow - libRow|
#     ( proximity ), then libRow ascending  ( pyEDM tieBreak rule )
#   - Simplex weights with the 1E-6 minimum-distance floor, projection
#     as the weight-normalized average of library target at neighbor+Tp
#
# SCOPE ( mandatory, tier 1 ) : default generative geometry only, i.e.
# libOverlap False, exclusionRadius 0, validLib empty, library large
# enough that knn finite neighbors always exist. Non-default geometry
# ( exclusion / validLib / straddling lib / deficiency ) is OUT of scope
# and MUST route to the pyEDM fallback ( see KernelEligible ). All
# compute is float32.

WEIGHT_FLOOR = np.float32( 1e-6 )   # pyEDM Project() minWeight floor

#----------------------------------------------------------------------
def KernelEligible( lib, exclusionRadius, validLib, libOverlap ):
    '''Precondition guard : True only for the default generative path.

    The kernel reproduces pyEDM only under the geometry GMN's default
    lib/pred produces. Anything else ( exclusion, validLib, straddling
    library, or a library too small for knn ) must defer to the pyEDM
    fallback rather than risk a silent divergence.

    Arguments:
        lib             : NodeLibrary. Frozen library for the node.
        exclusionRadius : int. Must be 0 for the kernel path.
        validLib        : list. Must be empty for the kernel path.
        libOverlap      : bool. Must be False ( pred beyond library ).

    Returns:
        bool : True if the kernel may handle this node, else defer.
    '''

    # Any non-default condition disqualifies the fast path.
    if exclusionRadius and exclusionRadius > 0 :
        return False
    if validLib is not None and len( validLib ) > 0 :
        return False
    if libOverlap :
        return False

    # Library must supply at least knn candidates.
    knn = lib.E + 1
    if lib.libEmbedded.shape[0] < knn :
        return False

    return True

#----------------------------------------------------------------------
def Neighbors( lib, predVec, knn ):
    '''Brute-force knn against the frozen library with the pyEDM tie rule.

    Computes Euclidean distances from predVec to every library embedding
    vector, then selects knn by the deterministic ordering
    ( distance, proximity |predRow - libRow|, libRow ). Because on the
    default path the single pred row sits beyond all library rows,
    proximity is monotonic in libRow and fully determines any distance
    tie ; the libRow key is a redundant final guarantee.

    Arguments:
        lib     : NodeLibrary. Frozen library ( libEmbedded, libRowIndex ).
        predVec : ndarray ( nCols*E, ) float32. Current pred embedding.
        knn     : int. Neighbors to select ( E + 1 default ).

    Returns:
        ( nbrRows, nbrDist ) : ndarray ( knn, ) of ORIGINAL library row
        indices ( for target gather and parity ) and their float32
        Euclidean distances, ordered by the deterministic rule.
    '''

    # Euclidean distance predVec -> every library vector ( float32 ).
    diff = lib.libEmbedded - predVec[ None, : ]
    dist = np.sqrt( np.einsum( 'ij,ij->i', diff, diff ) )
    dist = dist.astype( np.float32 )

    libRows = lib.libRowIndex                 # original data-row indices
    predRow = lib.predRow                     # current pred row index
    prox    = np.abs( predRow - libRows )     # proximity key

    # Deterministic lexsort : last key is primary. Order candidates by
    # distance, then proximity, then libRow ascending.
    order = np.lexsort( ( libRows, prox, dist ) )
    sel   = order[ :knn ]

    return libRows[ sel ], dist[ sel ]

#----------------------------------------------------------------------
def Project( nbrDist, nbrTarget ):
    '''Simplex projection : 1E-6-floored exponential weights ( pyEDM ).

    Reproduces Simplex.Project() for the default path ( all neighbors
    finite : no inf padding, no empty rows ). Weight = exp( -d / dMin )
    with dMin floored at 1E-6 ; projection = weighted average of the
    library target values at neighbor + Tp.

    Arguments:
        nbrDist   : ndarray ( knn, ) float32. Neighbor distances.
        nbrTarget : ndarray ( knn, ) float32. Library target at
                    neighbor + Tp ( precomputed in NodeLibrary ).

    Returns:
        ( projection, weights ) : float32 scalar and the ( knn, ) weight
        vector ( weights returned for the parity weight-layer check ).
    '''

    # Minimum distance, floored at 1E-6 to avoid divide-by-zero.
    dMin    = np.fmax( nbrDist.min(), WEIGHT_FLOOR )
    scaled  = nbrDist / dMin
    weights = np.exp( -scaled ).astype( np.float32 )

    wSum       = weights.sum()
    wSum_safe  = wSum if wSum > 0 else np.float32( 1.0 )

    projection = np.float32( ( weights * nbrTarget ).sum() / wSum_safe )

    return projection, weights

#----------------------------------------------------------------------
def Generate( lib ):
    '''One kernel generative step for a node : pred vector -> scalar.

    Assembles the pred embedding from the rolling buffer, finds knn,
    projects. Returns the scalar the GMN loop appends. Does NOT advance
    the rolling buffer : the caller commits all node outputs, then calls
    lib.Append() so every node advances against the same t-1 state.

    Arguments:
        lib : NodeLibrary. The node's frozen library + rolling buffer.

    Returns:
        float : the generated value for this node this timestep.
    '''

    knn              = lib.E + 1
    predVec          = lib.PredVector()
    nbrRows, nbrDist = Neighbors( lib, predVec, knn )

    # Gather library target at neighbor + Tp via the row -> position map.
    nbrTarget = lib.TargetAt( nbrRows )

    projection, _ = Project( nbrDist, nbrTarget )

    return float( projection )
