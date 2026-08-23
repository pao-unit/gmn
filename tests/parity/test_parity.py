#! /usr/bin/env python3
# pytest wrapper for the parity gate : regenerate fixtures against the
# installed pyEDM oracle, then run the float32 kernel candidate through
# the split contract. Marked 'pyedm' : the whole point is oracle parity.
#
# Kept separate from parity_runner.py ( the standalone CLI runner ) so the
# runner stays usable outside pytest.

import os
import sys
import tempfile

import pytest

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

#----------------------------------------------------------------------
# The knn 8 / exclusionRadius 20 guard fixture ( make_fixtures.py:481 )
# deliberately asks for more neighbors than the exclusion radius leaves
# available, so pyEDM's two FindNeighbors warnings are the expected
# result, not a defect. Silenced here rather than globally so the same
# warning still surfaces if it appears in any other test. The patterns
# use .* for the colons in the messages : a colon would be read as a
# warning-filter field separator.
@pytest.mark.pyedm
@pytest.mark.filterwarnings(
    'ignore:Simplex.*Fewer than knn.*neighbors outside exclusionRadius' )
@pytest.mark.filterwarnings(
    'ignore:Simplex.*no valid neighbors outside exclusionRadius' )
def test_kernel_parity_all_fixtures():
    '''All parity fixtures pass with the float32 kernel candidate.'''

    import make_fixtures
    import parity_runner
    from kernel_candidate import KernelCandidate

    with tempfile.TemporaryDirectory() as fixDir :
        # Regenerate fixtures against the installed oracle.
        make_fixtures.FamilyA( fixDir )
        make_fixtures.TieFamilyA( fixDir )
        make_fixtures.GuardFamilyA( fixDir )
        make_fixtures.FamilyB( fixDir )
        make_fixtures.FamilyC( fixDir )

        import json
        header = {
            'oracleVersion' : make_fixtures.ORACLE_VERSION,
            'oracleCommit'  : make_fixtures.ORACLE_COMMIT,
            'relTol'        : make_fixtures.REL_TOL,
            'weightFloor'   : make_fixtures.WEIGHT_FLOOR,
        }
        with open( os.path.join( fixDir, 'manifest.json' ), 'w' ) as f :
            json.dump( header, f )

        # Run the kernel candidate through every fixture.
        import glob
        candidate = KernelCandidate()
        relTol    = make_fixtures.REL_TOL

        import numpy as np
        failures = []

        for mf in sorted( glob.glob( os.path.join( fixDir, '*.json' ) ) ) :
            if mf.endswith( 'manifest.json' ) :
                continue

            meta   = json.load( open( mf ) )
            arrays = dict( np.load( mf[ :-5 ] + '.npz' ) )
            name   = meta[ 'name' ]

            if meta.get( 'kind' ) == 'trajectory' :
                ok, fails = parity_runner.CheckTrajectory( name, meta,
                                                           arrays, relTol )
            else :
                ok, fails = parity_runner.CheckSingleStep(
                                name, meta, arrays, candidate, relTol )

            if not ok :
                failures.extend( fails )

        assert not failures, 'parity failures : ' + '; '.join( failures )

#----------------------------------------------------------------------
if __name__ == '__main__':
    test_kernel_parity_all_fixtures()
    print( 'PARITY TEST PASSED' )
