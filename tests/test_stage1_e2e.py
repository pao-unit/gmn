#! /usr/bin/env python3
# Stage 1 end-to-end acceptance : run LOCALLY where pyEDM and the
# ABCD_Test network data / config are available. Closes T3 (parallel ==
# serial end-to-end) and exercises the vectorized commit path.
#
# Usage:
#   python3 tests/test_stage1_e2e.py -i <config.cfg> [-c CORES]
# Uses the same CLI as apps/Run.py; runs the config three ways and
# compares DataOut frame-for-frame.

import sys
from copy import deepcopy

import gmn
from gmn.CLI_Parser   import ParseCmdLine
from gmn.ConfigParser import ReadConfig

#----------------------------------------------------------------------
def RunBackend( args, parameters, backend, chunks = None ):
    '''Instantiate a fresh GMN and return its generated DataOut.'''

    a         = deepcopy( args )
    a.backend = backend
    a.chunks  = chunks
    a.Plot    = False        # suppress plotting during comparison
    a.StatePlot = False

    G = gmn.GMN( a, deepcopy( parameters ) )
    G.Generate()

    return G.DataOut

#----------------------------------------------------------------------
def Compare( label, a, b ):
    '''Assert two DataOut frames are identical; report on failure.'''

    same = a.equals( b )

    if not same :
        # Fall back to a numeric closeness report for diagnosis.
        import numpy as np
        diff = ( a.select_dtypes( 'number' ).to_numpy() -
                 b.select_dtypes( 'number' ).to_numpy() )
        print( f'FAIL {label}: max abs diff = {np.nanmax( np.abs(diff) )}' )
        return False

    print( f'PASS {label}: DataOut identical' )
    return True

#----------------------------------------------------------------------
def main():
    args       = ParseCmdLine()
    parameters = ReadConfig( args )

    # Reference : serial backend.
    serial = RunBackend( args, parameters, 'serial' )

    ok = True
    # T3 : parallel with the default chunk count must match serial.
    par = RunBackend( args, parameters, 'parallel' )
    ok &= Compare( 'parallel(default) vs serial', serial, par )

    # And with a couple of explicit chunk counts.
    for cc in [ 1, 3 * max(1, args.cores) ]:
        parN = RunBackend( args, parameters, 'parallel', chunks = cc )
        ok  &= Compare( f'parallel(chunks={cc}) vs serial', serial, parN )

    print( 'STAGE 1 E2E:', 'PASS' if ok else 'FAIL' )
    sys.exit( 0 if ok else 1 )

#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
