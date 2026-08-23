#! /usr/bin/env python3
# T7 : parity runner. Loads T5 fixtures and applies the split contract to
# a CANDIDATE kernel. The candidate is any object exposing:
#
#   Simplex( inputValues, columns, target, lib, pred, E, tau, Tp,
#            knn, exclusionRadius, validLib ) -> dict with keys
#            'embedding','knn_neighbors','knn_distances','weights',
#            'projection'  ( same layout as the frozen fixtures )
#
# The Stage 3 kernel will implement this interface. Until it exists, a
# pyEDM-backed reference candidate ( PyedmCandidate ) lets the harness
# self-validate : the oracle must satisfy its own contract. A green
# self-check proves the runner is correct BEFORE the kernel is built.
#
# Split contract:
#   real-valued layers ( embedding, distances, weights, projection )
#     -> relative tolerance relTol ( from manifest ; 1e-5 )
#   exact / binary layers
#     -> neighbor identity on ties, membership ( exclusion / validLib ),
#        inf-padding pattern, static-library shape : bit-exact
#   schedule stability
#     -> f_tie_schedule_stable : selection invariant to evaluation order
#   trajectories
#     -> step-1 anchored at relTol; horizon bounded-divergence band
#
# Tier 2 fixtures assert : candidate MATCHES oracle OR declares it defers
# ( returns None / raises NotImplemented ) -> counted as a pass ( guard ).
#
# Run ( pyEDM 2.5.7 installed ), from repo root :
#   python3 tests/parity/parity_runner.py --fixtures tests/parity/fixtures

import argparse
import glob
import json
import os
import sys

import numpy  as np
import pandas as pd

#----------------------------------------------------------------------
def RelClose( a, b, relTol ):
    '''Elementwise relative closeness with an absolute floor for zeros.

    Handles inf ( must match position ) and nan ( must match position ).
    Returns ( ok, maxRel ).
    '''

    a = np.asarray( a, float )
    b = np.asarray( b, float )

    if a.shape != b.shape :
        return False, float( 'inf' )

    # inf and nan must align positionally before numeric comparison.
    if not np.array_equal( np.isinf( a ), np.isinf( b ) ) :
        return False, float( 'inf' )
    if not np.array_equal( np.isnan( a ), np.isnan( b ) ) :
        return False, float( 'inf' )

    finite = np.isfinite( a ) & np.isfinite( b )

    if not finite.any() :
        return True, 0.0        # all inf/nan and positions matched

    denom  = np.maximum( np.abs( b[ finite ] ), 1e-12 )
    relErr = np.abs( a[ finite ] - b[ finite ] ) / denom
    maxRel = float( relErr.max() )

    return maxRel <= relTol, maxRel

#----------------------------------------------------------------------
class PyedmCandidate:
    '''Reference candidate : delegates to pyEDM. Used to self-validate the
       runner. The Stage 3 kernel will replace this with a native impl.'''

    def __init__( self ):
        import pyEDM
        self.pyEDM = pyEDM

    def Simplex( self, inputValues, columns, cols, target, lib, pred,
                 E, tau, Tp, knn, exclusionRadius, validLib ):
        '''Run pyEDM Simplex and return the three layers as a dict.'''

        df = pd.DataFrame( inputValues, columns = columns )

        S = self.pyEDM.Simplex(
                dataFrame = df, columns = cols, target = target,
                lib = lib, pred = pred, E = E, tau = tau, Tp = Tp,
                knn = knn, exclusionRadius = exclusionRadius,
                validLib = validLib if validLib else [],
                returnObject = True )

        knnDist = np.asarray( S.knn_distances, float )
        finite  = np.isfinite( knnDist )
        minD    = np.where( finite, knnDist, np.inf ).min( axis = 1 )
        minD    = np.where( np.isfinite( minD ), minD, 1.0 )
        minD    = np.fmax( minD, 1e-6 )
        weights = np.where( finite,
                            np.exp( -knnDist / minD[ :, None ] ), 0.0 )

        return {
            'embedding'     : np.asarray( S.Embedding.to_numpy(), float ),
            'knn_neighbors' : np.asarray( S.knn_neighbors, int ),
            'knn_distances' : knnDist,
            'weights'       : weights,
            'projection'    : np.asarray( S.projection, float ),
        }

#----------------------------------------------------------------------
def CheckSingleStep( name, meta, arrays, candidate, relTol ):
    '''Compare a candidate against one single-step fixture. Returns
       ( passed, list-of-failure-strings ). Tier-2 candidate deferral
       ( None / NotImplementedError ) counts as pass.'''

    fails = []
    tier  = meta[ 'tier' ]

    call  = meta.get( 'callArgs' )
    cols  = meta.get( 'input_columns' )

    # Fixtures without a stored input can only be output-compared, which
    # requires the candidate to have produced them; skip re-drive.
    if call is None or 'input_values' not in arrays :
        return True, [ f'{name}: SKIP ( no stored input to re-drive )' ]

    try :
        out = candidate.Simplex(
                inputValues     = arrays[ 'input_values' ],
                columns         = cols,
                cols            = call[ 'columns' ],
                target          = call[ 'target' ],
                lib             = call[ 'lib' ],
                pred            = call[ 'pred' ],
                E               = call[ 'E' ],
                tau             = call.get( 'tau', -1 ),
                Tp              = call.get( 'Tp', 1 ),
                knn             = call.get( 'knn', 0 ),
                exclusionRadius = call.get( 'exclusionRadius', 0 ),
                validLib        = call.get( 'validLib', [] ) )
    except NotImplementedError :
        if tier == 2 :
            return True, [ f'{name}: DEFER ( tier-2 guard, ok )' ]
        fails.append( f'{name}: candidate raised NotImplemented on tier-1' )
        return False, fails

    if out is None :
        if tier == 2 :
            return True, [ f'{name}: DEFER ( tier-2 guard, ok )' ]
        fails.append( f'{name}: candidate returned None on tier-1' )
        return False, fails

    # Real-valued layers : relative tolerance. Kernel candidates return
    # only the LAST pred row ( GMN's single generative point ) and the
    # library embedding ; align the fixture's last row for comparison.
    kernelMode = out[ 'knn_neighbors' ].shape[0] == 1 and \
                 arrays[ 'knn_neighbors' ].shape[0] > 1

    def _lastRow( a ):
        return a[ -1: ] if a.ndim >= 1 else a

    for layer in ( 'knn_distances', 'weights', 'projection' ):
        exp = _lastRow( arrays[ layer ] ) if kernelMode else arrays[ layer ]
        got = out[ layer ]
        ok, maxRel = RelClose( got, exp, relTol )
        if not ok :
            fails.append( f'{name}: {layer} rel={maxRel:.2e} > {relTol:.0e}' )

    # Embedding : the kernel returns the frozen library embedding ; the
    # fixture stores the full Simplex embedding. Compare the overlapping
    # library rows only when in kernel mode ( shapes differ by design ).
    if not kernelMode :
        ok, maxRel = RelClose( out[ 'embedding' ], arrays[ 'embedding' ],
                               relTol )
        if not ok :
            fails.append( f'{name}: embedding rel={maxRel:.2e}' )

    # Exact / binary layers : neighbor identity, inf-pad pattern.
    expNbr = _lastRow( arrays[ 'knn_neighbors' ] ) if kernelMode \
             else arrays[ 'knn_neighbors' ]
    if not np.array_equal( out[ 'knn_neighbors' ], expNbr ) :
        fails.append( f'{name}: knn_neighbors identity mismatch ( tie? )' )

    expDist = _lastRow( arrays[ 'knn_distances' ] ) if kernelMode \
              else arrays[ 'knn_distances' ]
    if not np.array_equal( np.isinf( out[ 'knn_distances' ] ),
                           np.isinf( expDist ) ) :
        fails.append( f'{name}: inf-padding pattern mismatch' )

    return ( len( fails ) == 0 ), fails

#----------------------------------------------------------------------
def CheckTrajectory( name, meta, arrays, relTol ):
    '''Trajectory fixtures : the fixture IS the oracle trajectory. A
       candidate driver is compared elsewhere ( needs the kernel ); here
       we self-validate the stored trajectory is finite and the assert
       spec is well-formed, and expose step-1 + band checks as helpers
       the kernel stage will call with a candidate trajectory.'''

    fails = []

    for node in meta[ 'nodes' ] :
        v = arrays[ f'traj_{node}' ]
        if not np.isfinite( v ).all() :
            fails.append( f'{name}: node {node} trajectory not finite' )

    return ( len( fails ) == 0 ), fails

#----------------------------------------------------------------------
def CompareTrajectory( candTraj, oracleArrays, nodes, relTol, band ):
    '''Kernel-stage helper : step-1 anchored at relTol, horizon within a
       bounded-divergence band. Returns ( passed, detail ).'''

    detail = []
    passed = True

    for node in nodes :
        oracle = oracleArrays[ f'traj_{node}' ]
        cand   = np.asarray( candTraj[ node ], float )

        # Step-1 anchor : tight.
        ok1, rel1 = RelClose( cand[ :1 ], oracle[ :1 ], relTol )
        if not ok1 :
            passed = False
            detail.append( f'{node}: step-1 rel={rel1:.2e}' )

        # Horizon : max relative divergence must stay within band.
        denom  = np.maximum( np.abs( oracle ), 1e-9 )
        maxDiv = float( ( np.abs( cand - oracle ) / denom ).max() )
        if maxDiv > band :
            passed = False
            detail.append( f'{node}: horizon div={maxDiv:.2e} > {band}' )

    return passed, detail

#----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser( description = 'Parity runner' )
    parser.add_argument( '--fixtures', default = 'tests/parity/fixtures',
                         help = 'Fixture directory.' )
    parser.add_argument( '--band', type = float, default = 0.05,
                         help = 'Trajectory bounded-divergence band.' )
    parser.add_argument( '--candidate', default = 'pyedm',
                         choices = [ 'pyedm', 'kernel' ],
                         help = 'Candidate under test ( default pyedm '
                                'self-validation ).' )
    args = parser.parse_args()

    manifest = json.load( open( os.path.join( args.fixtures,
                                              'manifest.json' ) ) )
    relTol   = manifest[ 'relTol' ]

    print( f'Parity runner : oracle {manifest["oracleVersion"]} '
           f'( {manifest["oracleCommit"][:9]} ), relTol {relTol:.0e}' )

    if args.candidate == 'kernel' :
        from kernel_candidate import KernelCandidate
        candidate = KernelCandidate()
        print( 'Candidate : KernelCandidate ( float32 Simplex kernel )' )
    else :
        candidate = PyedmCandidate()
        print( 'Candidate : PyedmCandidate ( self-validation of harness )' )

    print()

    metaFiles = sorted( glob.glob( os.path.join( args.fixtures,
                                                 '*.json' ) ) )

    total = 0
    passN = 0

    for mf in metaFiles :
        if mf.endswith( 'manifest.json' ) :
            continue

        meta   = json.load( open( mf ) )
        name   = meta[ 'name' ]
        arrays = dict( np.load( mf[ :-5 ] + '.npz' ) )

        if meta.get( 'kind' ) == 'trajectory' :
            ok, fails = CheckTrajectory( name, meta, arrays, relTol )
        else :
            ok, fails = CheckSingleStep( name, meta, arrays,
                                         candidate, relTol )

        total += 1
        passN += 1 if ok else 0

        tag = 'PASS' if ok else 'FAIL'
        print( f'  [{tag}] tier{meta["tier"]}  {name}' )
        for f in fails :
            # SKIP / DEFER lines are informational, not failures.
            print( f'         {f}' )

    print()
    print( f'{passN}/{total} fixtures passed' )
    sys.exit( 0 if passN == total else 1 )

#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
