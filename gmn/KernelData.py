
# Python distribution modules
import numpy as np

# T8 : static per-node data model for the specialized Simplex kernel.
#
# The GMN generative library is FROZEN at predictionStart ( libEnd_i is
# constant ), so each node's embedded library is a fixed-size float32
# array built ONCE and never grown : memory is O( N_lib x E x nCols ),
# constant across the horizon, not O( horizon ). Only a short rolling
# raw-history tail of length ( E - 1 ) * |tau| per column is carried to
# assemble the next prediction embedding vector each step.
#
# Embedding matches pyEDM 2.5.7 Embed() exactly : per column, delay rows
# are pandas shift( periods = [ 0, -tau, -2 tau, ... ] ), producing
# columns interleaved by shift period, with |tau| * ( E - 1 ) leading
# nan rows. Reproduced here with numpy so the kernel needs no pandas /
# pyEDM per step. Storage is float32; compute is float32 ( see design ).

#----------------------------------------------------------------------
def EmbedColumns( values, E, tau ):
    '''Takens delay embedding of raw columns, matching pyEDM Embed().

    For nCols input columns the output has nCols * E columns, ordered
    the way pandas DataFrame.shift( periods = shiftVec ) interleaves
    them : for each input column, its E delay copies at shift periods
    [ 0, -tau, -2 tau, ... ]. Leading ( E - 1 ) * |tau| rows contain nan
    ( insufficient history ), exactly as pyEDM.

    Arguments:
        values : ndarray ( nRows, nCols ) float. Raw column values in
                 the order the node's columns are listed.
        E      : int >= 1. Embedding dimension.
        tau    : int != 0. Delay ( GMN default -1 ).

    Returns:
        ndarray ( nRows, nCols * E ) float32. Delay-embedded block with
        leading nan rows where history is insufficient.
    '''

    values  = np.asarray( values, dtype = np.float32 )
    nRows   = values.shape[0]
    nCols   = values.shape[1]

    # Shift periods : pandas shift index is opposite the tau convention,
    # so shiftVec = [ 0, -tau, -2 tau, ... ] ( E entries ).
    shiftVec = [ i for i in range( 0, int( E * ( -tau ) ), -tau ) ]

    # Output column order matches pandas DataFrame[cols].shift( periods =
    # shiftVec ) : DELAY-major, column-minor. For shiftVec [0,1,2] and
    # columns [A,B] pyEDM yields A(t-0),B(t-0),A(t-1),B(t-1),A(t-2),B(t-2)
    # i.e. outer loop over shift period, inner loop over column.
    out = np.full( ( nRows, nCols * E ), np.nan, dtype = np.float32 )

    k = 0

    for s in shiftVec :
        for c in range( nCols ):
            # Positive shift period s moves values DOWN by s rows ( a row
            # receives the value from s rows earlier ), leaving s leading
            # nan. Matches pandas shift( periods = s ) for s >= 0.
            if s == 0 :
                out[ :, k ] = values[ :, c ]
            else :
                out[ s:, k ] = values[ :-s, c ]
            k += 1

    return out

#----------------------------------------------------------------------
class NodeLibrary:
    '''Frozen embedded library + rolling buffer for one kernel node.

    Built once at setup from the node's raw data through predictionStart.
    Holds the static embedded library ( float32 ), the aligned target
    vector, and a rolling raw-history tail used to assemble each step's
    prediction embedding without re-embedding the whole series.
    '''

    #------------------------------------------------------------------
    def __init__( self, rawValues, targetValues, E, tau, libEnd_i,
                  Tp = 1 ):
        '''Freeze the library from raw column data.

        Arguments:
            rawValues    : ndarray ( nRows, nCols ) float. Node input
                           columns ( predecessors + self ), column order
                           fixed for the run.
            targetValues : ndarray ( nRows, ) float. The node's target
                           column ( self ) aligned to rawValues rows.
            E, tau, Tp   : Simplex parameters.
            libEnd_i     : int. Frozen library end ( predictionStart ).
        '''

        self.E        = E
        self.tau      = tau
        self.Tp       = Tp
        self.libEnd_i = libEnd_i
        self.nCols    = np.asarray( rawValues ).shape[1]
        self.span     = ( E - 1 ) * abs( tau )   # rolling tail length

        # Embed the full series once, then slice the frozen library rows.
        embedded = EmbedColumns( rawValues, E, tau )

        # Library rows : valid embedded rows up to libEnd_i. Rows with
        # leading nan ( first span rows ) are excluded, matching pyEDM's
        # library index validity. The trailing Tp rows are ALSO excluded
        # because their target-at-neighbor+Tp would fall outside the
        # library ( pyEDM drops these from lib_i ).
        firstValid = self.span
        lastValid  = libEnd_i - Tp
        libRows     = np.arange( firstValid, lastValid )

        self.libEmbedded = np.ascontiguousarray(
                               embedded[ libRows ], dtype = np.float32 )

        # Target aligned to library rows, shifted by Tp for projection :
        # Simplex projects target at neighbor + Tp.
        target = np.asarray( targetValues, dtype = np.float32 )
        self.libTarget    = target[ libRows ]
        self.libTargetTp  = target[ libRows + Tp ]
        self.libRowIndex  = libRows        # original indices for tie key

        # Map original library row index -> position in libRowIndex, so
        # the kernel can gather target-at-neighbor+Tp by original row.
        self.rowToPos = { int( r ) : i for i, r in enumerate( libRows ) }

        # Current prediction row index ( original data-row of the pred
        # point ). Starts at the last seeded row; advances on Append so
        # the proximity tie key |predRow - libRow| stays correct.
        self.predRow = int( np.asarray( rawValues ).shape[0] - 1 )

        # Rolling raw tail : last ( span + 1 ) rows per column, enough to
        # build the next prediction embedding vector.
        self.rollBuffer = np.asarray(
                              rawValues[ -( self.span + 1 ): ],
                              dtype = np.float32 ).copy()

    #------------------------------------------------------------------
    def PredVector( self ):
        '''Assemble the current prediction embedding vector ( 1 x nCols*E )
           from the rolling buffer tail, matching the library embedding.

        Returns:
            ndarray ( nCols * E, ) float32 : the pred point in embedded
            space, ready for distance computation against libEmbedded.
        '''

        # Embed the small tail; the last row is the current pred vector.
        embeddedTail = EmbedColumns( self.rollBuffer, self.E, self.tau )

        return embeddedTail[ -1 ]

    #------------------------------------------------------------------
    def Append( self, newRawRow ):
        '''Advance the rolling buffer by one generated timestep.

        The library does NOT change ( frozen ); only the rolling tail
        rolls forward so the next PredVector() reflects the new point.

        Arguments:
            newRawRow : ndarray ( nCols, ) float. New raw values for this
                        node's input columns at the generated timestep.
        '''

        row = np.asarray( newRawRow, dtype = np.float32 ).reshape( 1, -1 )

        # Roll : drop oldest, append newest; keep length span + 1.
        self.rollBuffer = np.vstack(
                              [ self.rollBuffer[ 1: ], row ] )

        # Advance the prediction row index : the pred point moves one step
        # past the ( frozen ) library, keeping proximity keys correct.
        self.predRow += 1

    #------------------------------------------------------------------
    def TargetAt( self, nbrRows ):
        '''Gather library target-at-neighbor+Tp for original row indices.

        Arguments:
            nbrRows : ndarray ( knn, ) int. Original library row indices
                      returned by the kernel neighbor search.

        Returns:
            ndarray ( knn, ) float32 : libTargetTp at those rows.
        '''

        pos = np.array( [ self.rowToPos[ int( r ) ] for r in nbrRows ],
                        dtype = int )

        return self.libTargetTp[ pos ]
