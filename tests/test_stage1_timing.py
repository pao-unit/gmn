#! /usr/bin/env python3
# Stage 1 timing harness (T4) : measure parallel speedup over serial on
# the real pyEDM path, and expose the GIL ceiling that the Stage 3 kernel
# must raise. Run LOCALLY where pyEDM and the network data are available,
# from the directory that makes the config's relative paths resolve
# (e.g. config/):
#
#   python3 ../tests/test_stage1_timing.py -i default.cfg -c 5
#
# Options (parsed after the standard GMN args are consumed):
#   --reps  N   repetitions per configuration, min time reported (default 3)
#   --sweep     sweep cores 1,2,4,8,... up to -c and report scaling
#
# Correctness is already covered by test_stage1_e2e.py; this script only
# times. It also re-verifies parallel == serial once, so a timing run can
# never silently report a speedup for a wrong result.

import sys
import time
from copy  import deepcopy

import gmn
from gmn.CLI_Parser   import ParseCmdLine
from gmn.ConfigParser import ReadConfig

#----------------------------------------------------------------------
def RunOnce( args, parameters, backend, cores, chunks = None ):
    '''Run one Generate() and return ( DataOut, wallSeconds ).'''

    a           = deepcopy( args )
    a.backend   = backend
    a.cores     = cores
    a.chunks    = chunks
    a.Plot      = False        # suppress plotting during timing
    a.StatePlot = False
    a.verbose   = False
    a.DEBUG     = False

    G = gmn.GMN( a, deepcopy( parameters ) )

    # Wall clock around the generative loop only ( construction excluded ).
    start = time.perf_counter()
    G.Generate()
    wall  = time.perf_counter() - start

    return G.DataOut, wall

#----------------------------------------------------------------------
def BestOf( args, parameters, backend, cores, reps, chunks = None ):
    '''Return ( DataOut, minWall ) over reps runs : min rejects noise.'''

    best   = None
    frame  = None

    for _ in range( reps ):
        out, wall = RunOnce( args, parameters, backend, cores, chunks )
        if best is None or wall < best :
            best  = wall
            frame = out

    return frame, best

#----------------------------------------------------------------------
def PopFlag( name, cast, default ):
    '''Extract a private flag from argv so ParseCmdLine ignores it.'''

    if name in sys.argv :
        i   = sys.argv.index( name )
        val = cast( sys.argv[ i + 1 ] ) if cast is not int or \
              sys.argv[ i + 1 ].isdigit() else default
        del sys.argv[ i : i + 2 ]
        return val
    return default

#----------------------------------------------------------------------
def PopBool( name ):
    '''Extract a private boolean flag from argv.'''

    if name in sys.argv :
        sys.argv.remove( name )
        return True
    return False

#----------------------------------------------------------------------
def main():
    # Pull private flags before the standard GMN parser sees argv.
    reps  = PopFlag( '--reps', int, 3 )
    sweep = PopBool( '--sweep' )

    args       = ParseCmdLine()
    parameters = ReadConfig( args )
    maxCores   = max( 1, args.cores )

    steps = parameters.predictionLength
    print( f'# nodes are read from the network; horizon = {steps} steps' )
    print( f'# reps per config = {reps} ( min wall reported )' )
    print()

    # Serial baseline.
    serialFrame, serialWall = BestOf( args, parameters, 'serial',
                                      cores = 1, reps = reps )
    print( f'serial            : {serialWall:8.3f} s   (baseline)' )

    # Core sweep or single parallel point.
    coreList = []
    if sweep :
        c = 1
        while c < maxCores :
            coreList.append( c )
            c *= 2
        coreList.append( maxCores )
    else :
        coreList = [ maxCores ]

    prevWall = serialWall

    for c in coreList :
        frame, wall = BestOf( args, parameters, 'parallel',
                              cores = c, reps = reps )

        # Guard : a timing run must never report speedup for a wrong result.
        assert frame.equals( serialFrame ), \
            f'parallel(cores={c}) DataOut != serial : timing invalid'

        speedup = serialWall / wall if wall > 0 else float( 'nan' )
        margApp = prevWall  / wall if wall > 0 else float( 'nan' )
        print( f'parallel cores={c:<3d}: {wall:8.3f} s   '
               f'speedup x{speedup:5.2f}   (vs prev x{margApp:4.2f})' )
        prevWall = wall

    print()
    print( '# Speedup that plateaus below cores indicates the GIL ceiling' )
    print( '# of the pyEDM path : the baseline Stage 3 kernel must raise.' )

#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
