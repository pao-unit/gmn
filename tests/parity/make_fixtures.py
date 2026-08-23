#! /usr/bin/env python3
# T5 : generate parity fixtures for the specialized Simplex kernel.
#
# Oracle : pyEDM 2.5.7 at commit 4ad6b04 ( deterministic tie ordering +
# nan masking ). Each fixture freezes a ( library data, parameter tuple,
# pred geometry ) input plus the oracle outputs at three layers captured
# from Simplex( returnObject = True ) : Embedding, knn_neighbors +
# knn_distances, projection. Weights are DERIVED from knn_distances via
# the Project convention ( 1E-6 floor ) and stored for the weight-layer
# assertion. Every fixture also records tieBreak and libOverlap so it is
# self-describing about the code path it exercised.
#
# All capture is single-step Simplex ( generateSteps unused ). Family C
# trajectories are produced by a GMN-faithful feedback loop, NOT by
# Simplex( generateSteps = N ).
#
# Fixtures are tiered:
#   tier 1 ( mandatory ) : default generative geometry the kernel must
#                          reproduce : libOverlap False, exclusionRadius 0,
#                          validLib empty, deterministic tie rule.
#   tier 2 ( guard )     : exclusion / validLib / deficiency. GMN's default
#                          path cannot produce these; kernel must match OR
#                          defer to pyEDM fallback. Marked tier=2.
#
# Run ( with pyEDM 2.5.7 installed ) from the repo root :
#   python3 tests/parity/make_fixtures.py --out tests/parity/fixtures

import argparse
import json
import os

import numpy  as np
import pandas as pd

import pyEDM

# Pinned oracle commit : recorded in the manifest header. Regeneration
# under a different pyEDM must be a visible, deliberate event.
ORACLE_VERSION = pyEDM.__version__
ORACLE_COMMIT  = '4ad6b049951af5052b00af21985acc74902d9bf8'

# Contract constants ( see design thread ).
REL_TOL     = 1e-5     # real-valued layer tolerance
WEIGHT_FLOOR = 1e-6    # Project() minWeight floor : fmax( minDist, 1E-6 )

#----------------------------------------------------------------------
def DeriveWeights( knnDistances ):
    '''Reproduce Project() weights from knn_distances ( 2.5.7 ).

    Mirrors Simplex.Project : min over FINITE neighbors, floor at 1E-6,
    scale, exp( -scaled ), padding ( inf ) slots forced to weight 0.

    Arguments:
        knnDistances : ndarray ( N_pred, k ). inf marks padding slots.

    Returns:
        ndarray ( N_pred, k ) of weights matching the oracle convention.
    '''

    finite = np.isfinite( knnDistances )

    # Minimum distance per row over finite slots only.
    minDist = np.where( finite, knnDistances, np.inf ).min( axis = 1 )
    minDist = np.where( np.isfinite( minDist ), minDist, 1.0 )
    minDist = np.fmax( minDist, WEIGHT_FLOOR )

    scaled  = knnDistances / minDist[ :, None ]
    weights = np.exp( -scaled )

    # Padding ( inf-distance ) slots contribute exactly 0.
    weights = np.where( finite, weights, 0.0 )

    return weights

#----------------------------------------------------------------------
def CaptureSimplex( data, columns, target, lib, pred, E, tau = -1,
                    Tp = 1, knn = 0, exclusionRadius = 0, validLib = [] ):
    '''Run single-step Simplex oracle and capture the three layers.

    Returns a dict of frozen arrays + captured path flags. generateSteps
    is left at its 0 default ( unused ), matching GMN's per-step call.

    Arguments mirror the Simplex API for the parameters fixtures vary.
    '''

    S = pyEDM.Simplex( dataFrame       = data,
                       columns         = columns,
                       target          = target,
                       lib             = lib,
                       pred            = pred,
                       E               = E,
                       tau             = tau,
                       Tp              = Tp,
                       knn             = knn,
                       exclusionRadius = exclusionRadius,
                       validLib        = validLib,
                       returnObject    = True )

    knnDist = np.asarray( S.knn_distances, dtype = float )
    knnNbr  = np.asarray( S.knn_neighbors, dtype = int )
    weights = DeriveWeights( knnDist )

    # Capture the LAST pred row : GMN's single generative prediction.
    capture = {
        'embedding'     : np.asarray( S.Embedding.to_numpy(), float ),
        'knn_neighbors' : knnNbr,
        'knn_distances' : knnDist,
        'weights'       : weights,
        'projection'    : np.asarray( S.projection, float ),
        'lib_i'         : np.asarray( S.lib_i, int ),
        'pred_i'        : np.asarray( S.pred_i, int ),
        'knn'           : int( S.knn ),
        'tieBreak'      : bool( S.tieBreak ),
        'libOverlap'    : bool( S.libOverlap ),
    }

    return capture

#----------------------------------------------------------------------
def SmoothFrame( names, rows, seed ):
    '''Coupled-oscillator frame : Time + one smooth column per name.'''

    rng = np.random.default_rng( seed )
    t   = np.arange( 1, rows + 1 )

    df = pd.DataFrame( { 'Time' : t } )

    for i, name in enumerate( names ):
        freq       = 0.05 + 0.03 * i
        df[ name ] = np.sin( freq * t ) + 0.01 * rng.standard_normal( rows )

    return df

#----------------------------------------------------------------------
def WriteFixture( outDir, name, tier, path, params, capture, notes,
                  inputFrame = None, callArgs = None ):
    '''Serialize one fixture : arrays to .npz, metadata to .json.

    inputFrame / callArgs, when given, let the parity runner re-drive a
    candidate kernel on the exact oracle input rather than only comparing
    frozen outputs. Stored as columns + values in the npz and a call spec
    in the json.
    '''

    npzPath  = os.path.join( outDir, f'{name}.npz'  )
    metaPath = os.path.join( outDir, f'{name}.json' )

    arrays = {
        'embedding'     : capture['embedding'],
        'knn_neighbors' : capture['knn_neighbors'],
        'knn_distances' : capture['knn_distances'],
        'weights'       : capture['weights'],
        'projection'    : capture['projection'],
        'lib_i'         : capture['lib_i'],
        'pred_i'        : capture['pred_i'],
    }

    # Optionally freeze the raw input so the runner can re-run a kernel.
    if inputFrame is not None :
        arrays[ 'input_values' ] = inputFrame.to_numpy( dtype = float )

    np.savez( npzPath, **arrays )

    # Metadata : parameters, path flags, tier, assertion notes.
    meta = {
        'name'       : name,
        'tier'       : tier,          # 1 mandatory, 2 guard
        'pathClass'  : path,          # 'default' | 'exclusion' | 'validLib'
        'params'     : params,
        'knn'        : capture['knn'],
        'tieBreak'   : capture['tieBreak'],
        'libOverlap' : capture['libOverlap'],
        'notes'      : notes,
    }

    # Record input columns + call spec so the runner can re-drive a kernel.
    if inputFrame is not None :
        meta[ 'input_columns' ] = list( inputFrame.columns )
    if callArgs is not None :
        meta[ 'callArgs' ] = callArgs

    with open( metaPath, 'w' ) as f :
        json.dump( meta, f, indent = 2 )

    flags = f"tier={tier} tieBreak={capture['tieBreak']} " \
            f"libOverlap={capture['libOverlap']} knn={capture['knn']}"
    print( f'  wrote {name:22s} [{flags}]' )

#----------------------------------------------------------------------
def FamilyA( outDir ):
    '''Synthetic minimal : one fixture per targeted behavior.'''

    print( '[Family A : synthetic minimal]' )

    # --- default clean path : baseline nearest neighbors, no tie ---
    df   = SmoothFrame( [ 'A', 'B', 'C' ], rows = 60, seed = 1 )
    N    = df.shape[0]
    cap  = CaptureSimplex( df, 'A B C', 'A', '1 50', f'{N-1} {N}', E = 3 )
    WriteFixture( outDir, 'f_default_clean', 1, 'default',
                  { 'E':3, 'tau':-1, 'Tp':1, 'lib':'1 50',
                    'pred':f'{N-1} {N}' }, cap,
                  'Baseline default generative geometry; mandatory.',
                  inputFrame = df,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'1 50', 'pred':f'{N-1} {N}',
                               'E':3, 'tau':-1, 'Tp':1 } )

    # --- zero-distance floor : duplicate the pred EMBEDDING in library ---
    # The embedded vector is a time-delay window ( E columns, tau step ),
    # so a duplicate must copy the WHOLE window, not one raw row. With
    # E=3, tau=-1 the pred embedding at row r uses raw rows r, r-1, r-2
    # of each column. Copy that 3-row window into an early library
    # location so the embedded library vector equals the pred vector and
    # d_min = 0, engaging the 1E-6 floor. ( Same embedded-space rule the
    # tie fixtures require. )
    dfz    = SmoothFrame( [ 'A', 'B', 'C' ], rows = 60, seed = 2 )
    E, tau = 3, -1
    span   = ( E - 1 ) * abs( tau )              # rows the window spans
    predR  = dfz.index[-1]                        # last row = pred point
    libR   = dfz.index[ 15 ]                      # target library row
    # Copy the full delay window [ r-span .. r ] for every column.
    for off in range( 0, span + 1 ):
        for col in [ 'A', 'B', 'C' ]:
            dfz.loc[ libR - off, col ] = dfz.loc[ predR - off, col ]
    Nz  = dfz.shape[0]
    capz = CaptureSimplex( dfz, 'A B C', 'A', '1 50', f'{Nz-1} {Nz}', E=3 )
    WriteFixture( outDir, 'f_zerodist_floor', 1, 'default',
                  { 'E':3, 'note':'duplicate pred row in library' }, capz,
                  '1E-6 weight floor must engage; weight-layer canary.',
                  inputFrame = dfz,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'1 50', 'pred':f'{Nz-1} {Nz}',
                               'E':3, 'tau':-1, 'Tp':1 } )

    print( '  ( tie + exclusion + validLib fixtures : see build notes )' )

#----------------------------------------------------------------------
def FamilyB( outDir ):
    '''Realistic single-step from bundled ABCD data if available.'''

    print( '[Family B : realistic single-step]' )

    abcd = 'data/TestData_ABCD.csv'
    if not os.path.exists( abcd ):
        print( '  SKIP : data/TestData_ABCD.csv not found '
               '( run from repo root )' )
        return

    df = pd.read_csv( abcd )
    N  = df.shape[0]
    # Realistic : large frozen library, single pred at the end.
    libEnd = min( 800, N - 2 )
    cap    = CaptureSimplex( df, 'A B C D', 'A',
                             f'1 {libEnd}', f'{N-1} {N}', E = 5 )
    WriteFixture( outDir, 'f_real_abcd', 1, 'default',
                  { 'E':5, 'lib':f'1 {libEnd}', 'pred':f'{N-1} {N}' }, cap,
                  'Realistic library size; validates 1e-5 sufficiency.',
                  inputFrame = df,
                  callArgs = { 'columns':'A B C D', 'target':'A',
                               'lib':f'1 {libEnd}', 'pred':f'{N-1} {N}',
                               'E':5, 'tau':-1, 'Tp':1 } )

#----------------------------------------------------------------------
def TieFrameProximity( rows = 60, tieRows = ( 15, 40 ) ):
    '''Frame with a float32-exact distance tie broken by PROXIMITY.

    Two library embedding vectors are a permutation pair (3,4,0..) and
    (4,3,0..) in column A : identical squared-difference addend multiset
    => bit-identical distance in float32 AND float64. Both lie below the
    single pred point ( zero vector ), so |predRow-libRow| differs and
    the secondary ( proximity ) key decides : the row closer to pred
    ( higher index ) wins. This is the mandatory default-path tie case.

    Arguments:
        rows    : int. Series length.
        tieRows : (int,int). (farther, closer) library rows for the tie.

    Returns:
        ( DataFrame, closerRow ) : closerRow is the expected winner.
    '''

    cols = ( 'A', 'B', 'C' )
    df   = pd.DataFrame( { 'Time' : np.arange( 1, rows + 1 ) } )

    # Base : large ramped values so all untouched embeddings are far.
    for j, c in enumerate( cols ):
        df[ c ] = 100.0 + j * 10 + np.arange( rows ) * 0.5

    # Pred window ( last E rows ) : zero embedding vector.
    for off in ( 0, 1, 2 ):
        for c in cols :
            df.loc[ df.index[ -1 - off ], c ] = 0.0

    # Permutation-pair tie in column A; zero B,C over both windows.
    r1, r2 = tieRows                       # r2 closer to pred than r1
    df.loc[ r1, 'A' ] = 3.0 ; df.loc[ r1-1, 'A' ] = 4.0
    df.loc[ r1-2, 'A' ] = 0.0
    df.loc[ r2, 'A' ] = 4.0 ; df.loc[ r2-1, 'A' ] = 3.0
    df.loc[ r2-2, 'A' ] = 0.0

    for off in ( 0, 1, 2 ):
        for c in ( 'B', 'C' ):
            df.loc[ r1 - off, c ] = 0.0
            df.loc[ r2 - off, c ] = 0.0

    return df, max( r1, r2 )

#----------------------------------------------------------------------
def TieFrameIndex( rows = 90, predRow = 45, offset = 15 ):
    '''Frame with a tie broken by the TERTIARY libRow key ( tier 2 ).

    Requires a STRADDLING library ( rows both above and below pred ) so
    the two tied rows are symmetric about pred : equal distance AND equal
    proximity, leaving libRow-ascending to break it ( lower index wins ).
    This geometry is non-default ( libOverlap True ), hence tier 2 : the
    kernel must match OR defer to fallback, not required for GMN's path.

    Returns:
        ( DataFrame, predRow, lowerRow ) : lowerRow is expected winner.
    '''

    cols = ( 'A', 'B', 'C' )
    df   = pd.DataFrame( { 'Time' : np.arange( 1, rows + 1 ) } )

    for j, c in enumerate( cols ):
        df[ c ] = 100.0 + j * 10 + np.arange( rows ) * 0.5

    # Pred window zero.
    for off in ( 0, 1, 2 ):
        for c in cols :
            df.loc[ predRow - off, c ] = 0.0

    # Identical embedded vector on both symmetric rows => equal distance;
    # symmetric offset => equal proximity; libRow breaks toward lower.
    rLo, rHi = predRow - offset, predRow + offset

    for r in ( rLo, rHi ):
        df.loc[ r, 'A' ] = 3.0 ; df.loc[ r-1, 'A' ] = 4.0
        df.loc[ r-2, 'A' ] = 0.0
        for off in ( 0, 1, 2 ):
            for c in ( 'B', 'C' ):
                df.loc[ r - off, c ] = 0.0

    return df, predRow, rLo

#----------------------------------------------------------------------
def TieFamilyA( outDir ):
    '''Tie fixtures : proximity ( tier 1 ) and index ( tier 2 ).'''

    print( '[Family A : tie fixtures]' )

    # --- f_tie_proximity : mandatory, secondary-key tie ---
    dfp, closer = TieFrameProximity()
    Np  = dfp.shape[0]
    cap = CaptureSimplex( dfp, 'A B C', 'A', '1 50', f'{Np-1} {Np}', E = 3 )
    # Confirm the tie is real before freezing.
    d = cap['knn_distances'][ -1 ]
    assert d[0] == d[1], 'proximity tie not equidistant'
    assert np.float32( d[0] ) == np.float32( d[1] ), 'tie not float32-exact'
    assert cap['knn_neighbors'][ -1 ][ 0 ] == closer, 'proximity winner wrong'
    WriteFixture( outDir, 'f_tie_proximity', 1, 'default',
                  { 'E':3, 'construction':'permutation-pair, cols A',
                    'expectFirst':int( closer ) }, cap,
                  'Distance tie broken by proximity ( closer row wins ); '
                  'mandatory. float32-exact by permutation-pair.',
                  inputFrame = dfp,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'1 50', 'pred':f'{Np-1} {Np}',
                               'E':3, 'tau':-1, 'Tp':1 } )

    # --- f_tie_schedule_stable : same data, flagged for order-invariance ---
    # ( runner re-evaluates kernel under varied worker/chunk order and
    #   asserts identical selection; the FIXTURE is the same capture. )
    WriteFixture( outDir, 'f_tie_schedule_stable', 1, 'default',
                  { 'E':3, 'reuses':'f_tie_proximity geometry',
                    'assert':'kernel selection invariant to schedule' },
                  cap,
                  'Tie pick must be identical across worker/chunk orders; '
                  'reads data keys only, never arrival order.',
                  inputFrame = dfp,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'1 50', 'pred':f'{Np-1} {Np}',
                               'E':3, 'tau':-1, 'Tp':1 } )

    # --- f_tie_index : tier-2 guard, tertiary libRow key ( straddle ) ---
    dfi, predRow, lower = TieFrameIndex()
    Ni  = dfi.shape[0]
    capi = CaptureSimplex( dfi, 'A B C', 'A', f'1 {Ni}',
                           f'{predRow} {predRow+1}', E = 3 )
    # The straddle capture has 2 pred rows; find the symmetric ( tied ) one.
    nbr = capi['knn_neighbors'] ; dst = capi['knn_distances']
    picked = None
    for rowsel in range( dst.shape[0] ):
        if dst[ rowsel, 0 ] > 0 and dst[ rowsel, 0 ] == dst[ rowsel, 1 ]:
            picked = rowsel
    assert picked is not None, 'no tie row found in straddle capture'
    assert nbr[ picked ][ 0 ] == lower, 'tertiary libRow winner wrong'
    WriteFixture( outDir, 'f_tie_index', 2, 'straddle',
                  { 'E':3, 'predRow':int( predRow ),
                    'expectFirst':int( lower ),
                    'note':'non-default straddling lib ( libOverlap True )' },
                  capi,
                  'TIER 2 : tertiary libRow key ( lower index wins ) only '
                  'reachable with straddling lib; kernel matches OR defers.',
                  inputFrame = dfi,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':f'1 {Ni}', 'pred':f'{predRow} {predRow+1}',
                               'E':3, 'tau':-1, 'Tp':1 } )

#----------------------------------------------------------------------
def GuardFamilyA( outDir ):
    '''Tier-2 guard fixtures : exclusion, validLib, deficiency.

    None fire on GMN's default generative geometry ( libOverlap False,
    exclusionRadius 0, validLib empty, library large ). They guard the
    boundary a future non-default lib/pred would cross : the kernel must
    reproduce them OR its precondition check must route the node to the
    pyEDM fallback. All are tier=2.
    '''

    print( '[Family A : tier-2 guards]' )

    # --- f_exclusion_radius : exclusion only bites with a straddling lib ---
    # ( rows near the pred index ). Radius removes neighbors within
    # +/-radius of the pred row, forcing farther selections.
    dfe = SmoothFrame( [ 'A', 'B', 'C' ], rows = 80, seed = 4 )
    Ne  = dfe.shape[0]
    predRow = 45
    cap = CaptureSimplex( dfe, 'A B C', 'A', f'1 {Ne}',
                          f'{predRow} {predRow+1}', E = 3,
                          exclusionRadius = 8 )
    # Verify exclusion actually moved neighbors outside the radius.
    pr  = cap['pred_i'][ -1 ]
    nbr = cap['knn_neighbors'][ -1 ]
    assert np.all( np.abs( nbr - pr ) > 8 ), 'exclusion did not apply'
    WriteFixture( outDir, 'f_exclusion_radius', 2, 'exclusion',
                  { 'E':3, 'exclusionRadius':8, 'predRow':int( predRow ),
                    'note':'straddling lib; libOverlap True' }, cap,
                  'TIER 2 : neighbors within +/-radius of pred excluded. '
                  'Unreachable on default path; kernel matches OR defers.',
                  inputFrame = dfe,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':f'1 {Ne}', 'pred':f'{predRow} {predRow+1}',
                               'E':3, 'tau':-1, 'Tp':1,
                               'exclusionRadius':8 } )

    # --- f_validlib_omit : invalid rows must never appear as neighbors ---
    dfv = SmoothFrame( [ 'A', 'B', 'C' ], rows = 60, seed = 5 )
    Nv  = dfv.shape[0]
    # First capture the default picks, then invalidate them.
    base = CaptureSimplex( dfv, 'A B C', 'A', '1 50', f'{Nv-1} {Nv}', E = 3 )
    invalid = [ int( x ) for x in base['knn_neighbors'][ -1 ][ :2 ] ]
    validLib = [ True ] * Nv
    for idx in invalid :
        validLib[ idx ] = False
    cap = CaptureSimplex( dfv, 'A B C', 'A', '1 50', f'{Nv-1} {Nv}',
                          E = 3, validLib = validLib )
    # The invalidated rows must be absent from the new neighbor set.
    picked = set( int( x ) for x in cap['knn_neighbors'][ -1 ] )
    assert not ( set( invalid ) & picked ), 'invalid row still selected'
    WriteFixture( outDir, 'f_validlib_omit', 2, 'validLib',
                  { 'E':3, 'invalidatedRows':invalid,
                    'note':'validLib excludes rows from candidacy' }, cap,
                  'TIER 2 : invalid library rows never appear as neighbors. '
                  'Exact membership assertion; kernel matches OR defers.',
                  inputFrame = dfv,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'1 50', 'pred':f'{Nv-1} {Nv}',
                               'E':3, 'tau':-1, 'Tp':1,
                               'validLib':validLib } )

    # --- f_deficient_inf_pad : exclusion leaves < knn valid neighbors ---
    # NB: knn > library size RAISES in pyEDM ( not padded ). The inf-pad
    # path is reached only when the library is adequate but exclusion
    # thins it below knn : surplus slots get inf distance, weight 0, and
    # empty predictions become nan ( 2.5.7 masking ).
    dfd = SmoothFrame( [ 'A', 'B', 'C' ], rows = 80, seed = 6 )
    Nd  = dfd.shape[0]
    cap = CaptureSimplex( dfd, 'A B C', 'A', '30 50',
                          f'{40} {41}', E = 3, knn = 8,
                          exclusionRadius = 20 )
    # Confirm inf padding is present in the captured distances.
    hasInf = np.isinf( cap['knn_distances'] ).any()
    assert hasInf, 'deficiency did not produce inf padding'
    # Padding slots must carry weight exactly 0.
    padMask = np.isinf( cap['knn_distances'] )
    assert np.all( cap['weights'][ padMask ] == 0.0 ), 'pad weight != 0'
    WriteFixture( outDir, 'f_deficient_inf_pad', 2, 'exclusion',
                  { 'E':3, 'knn':8, 'exclusionRadius':20,
                    'note':'exclusion thins lib below knn' }, cap,
                  'TIER 2 : deficient rows pad with inf distance, weight 0, '
                  'empty rows nan ( 2.5.7 masking ). knn>libSize RAISES, '
                  'not padded : kernel must match this boundary or defer.',
                  inputFrame = dfd,
                  callArgs = { 'columns':'A B C', 'target':'A',
                               'lib':'30 50', 'pred':'40 41',
                               'E':3, 'tau':-1, 'Tp':1, 'knn':8,
                               'exclusionRadius':20 } )

#----------------------------------------------------------------------
def GmnTrajectory( data0, nodeInputs, nodeCols, libEnd_i, E, steps,
                   tau = -1, Tp = 1 ):
    '''Reproduce GMN.Generate()'s loop to capture an oracle trajectory.

    Each step every node runs single-step Simplex against the SAME t-1
    data ( lastDataOut ), all projections form one new row committed
    together, then the row is appended and pred advances. The library
    stays frozen at libEnd_i. This is the ONLY correct oracle for Family
    C : Simplex( generateSteps = N ) is NOT used, because GMN never calls
    it that way.

    Arguments:
        data0      : DataFrame. Seed data ( Time + node columns ).
        nodeInputs : dict node -> [input columns] ( predecessors + self ).
        nodeCols   : list[str]. Node columns in commit order.
        libEnd_i   : int. Frozen library end index.
        E, steps, tau, Tp : Simplex / horizon parameters.

    Returns:
        ( trajDict, finalData ) : trajDict is node -> ndarray of values.
    '''

    data = data0.copy()
    traj = { n : [] for n in nodeCols }

    for _ in range( steps ):
        N    = data.shape[0]
        lib  = f'1 {libEnd_i}'
        pred = f'{N-1} {N}'

        # All nodes read the same t-1 data before any commit.
        newVals = {}

        for node in nodeCols :
            cols = ' '.join( nodeInputs[ node ] )
            S = pyEDM.Simplex( dataFrame = data, columns = cols,
                               target = node, lib = lib, pred = pred,
                               E = E, tau = tau, Tp = Tp,
                               returnObject = True )
            newVals[ node ] = float( np.asarray( S.projection )[ -1 ] )

        # Commit all node outputs as ONE new row ( vectorized commit ).
        newRow = data.iloc[ [ -1 ] ].copy()
        newRow[ 'Time' ] = data[ 'Time' ].iloc[ -1 ] + 1

        for node in nodeCols :
            newRow[ node ] = newVals[ node ]
            traj[ node ].append( newVals[ node ] )

        data = pd.concat( [ data, newRow ], ignore_index = True )

    return { n : np.array( v ) for n, v in traj.items() }, data

#----------------------------------------------------------------------
def WriteTrajectoryFixture( outDir, name, tier, traj, params, notes ):
    '''Serialize a trajectory fixture : per-node value arrays + meta.'''

    npzPath  = os.path.join( outDir, f'{name}.npz'  )
    metaPath = os.path.join( outDir, f'{name}.json' )

    # One array per node : the oracle generative trajectory.
    np.savez( npzPath, **{ f'traj_{k}' : v for k, v in traj.items() } )

    meta = {
        'name'   : name,
        'tier'   : tier,
        'kind'   : 'trajectory',
        'nodes'  : list( traj.keys() ),
        'steps'  : int( len( next( iter( traj.values() ) ) ) ),
        'params' : params,
        'assert' : { 'step1' : 'rel 1e-5 vs oracle',
                     'horizon' : 'bounded-divergence band; same basin' },
        'notes'  : notes,
    }

    with open( metaPath, 'w' ) as f :
        json.dump( meta, f, indent = 2 )

    print( f'  wrote {name:22s} [tier={tier} nodes={len(traj)} '
           f'steps={meta["steps"]}]' )

#----------------------------------------------------------------------
def FamilyC( outDir ):
    '''Generative-trajectory fixtures via the GMN-faithful loop.'''

    print( '[Family C : GMN-loop trajectories]' )

    # --- f_traj_singlenode : one self-predicting node, no coupling ---
    rows = 80
    t    = np.arange( 1, rows + 1 )
    df1  = pd.DataFrame( { 'Time' : t, 'A' : np.sin( 0.1 * t ) } )
    traj1, _ = GmnTrajectory( df1, { 'A' : [ 'A' ] }, [ 'A' ],
                              libEnd_i = 60, E = 3, steps = 30 )
    assert np.isfinite( traj1[ 'A' ] ).all(), 'single-node traj not finite'
    WriteTrajectoryFixture( outDir, 'f_traj_singlenode', 1, traj1,
                            { 'E':3, 'libEnd_i':60, 'steps':30,
                              'nodes':1 },
                            'Compounding baseline, no coupling. Step-1 '
                            'tight; horizon bounded-divergence.' )

    # --- f_traj_coupled : multi-input node with real feedback ---
    df2 = pd.DataFrame( { 'Time' : t,
                          'A' : np.sin( 0.10 * t ),
                          'B' : np.cos( 0.08 * t ),
                          'C' : np.sin( 0.05 * t ) + 0.3 * np.cos( 0.08*t ) } )
    inputs = { 'A' : [ 'A' ], 'B' : [ 'B' ], 'C' : [ 'C', 'A', 'B' ] }
    traj2, _ = GmnTrajectory( df2, inputs, [ 'A', 'B', 'C' ],
                              libEnd_i = 60, E = 3, steps = 25 )
    for n in ( 'A', 'B', 'C' ):
        assert np.isfinite( traj2[ n ] ).all(), f'coupled traj {n} not finite'
    WriteTrajectoryFixture( outDir, 'f_traj_coupled', 1, traj2,
                            { 'E':3, 'libEnd_i':60, 'steps':25,
                              'nodes':3, 'coupling':'C<-A,B,C' },
                            'Real feedback : subtle tie / zero-distance '
                            'errors surface here as drift. Integration '
                            'canary. Also guards static-library invariant.' )

#----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser( description = 'Make parity fixtures' )
    parser.add_argument( '--out', default = 'tests/parity/fixtures',
                         help = 'Fixture output directory.' )
    args = parser.parse_args()

    os.makedirs( args.out, exist_ok = True )

    # Manifest header : the single source of contract truth.
    header = {
        'oracleVersion' : ORACLE_VERSION,
        'oracleCommit'  : ORACLE_COMMIT,
        'relTol'        : REL_TOL,
        'weightFloor'   : WEIGHT_FLOOR,
        'exactLayers'   : [ 'neighbor_identity_on_ties',
                            'membership_exclusion_validLib',
                            'inf_padding_deficient_rows',
                            'static_library_shape' ],
        'tiers'         : { '1' : 'mandatory default generative path',
                            '2' : 'guard: kernel matches OR defers to '
                                  'pyEDM fallback' },
        'notes'         : 'All capture single-step Simplex; generateSteps '
                          'unused. Family C uses a GMN-faithful loop.',
    }

    with open( os.path.join( args.out, 'manifest.json' ), 'w' ) as f :
        json.dump( header, f, indent = 2 )

    print( f'Oracle pyEDM {ORACLE_VERSION} ( commit {ORACLE_COMMIT[:9]} )' )
    print( f'Writing fixtures to {args.out}/' )

    FamilyA( args.out )
    TieFamilyA( args.out )
    GuardFamilyA( args.out )
    FamilyB( args.out )
    FamilyC( args.out )

    print( 'done. ( tie / exclusion / validLib / trajectory fixtures are '
           'staged incrementally per the manifest tiers )' )

#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
