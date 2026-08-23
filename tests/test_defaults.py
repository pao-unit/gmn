# Operative-default assertions and an end-to-end CLI app test. These need
# no pyEDM ( they check flag defaults and run the app on the kernel
# default path ), so they are always-on.

import os
import subprocess
import sys

import pandas as pd

import gmn

#----------------------------------------------------------------------
def test_default_backend_serial():
    '''Default dispatch backend is serial.'''

    args = gmn.CLI_Parser.ParseCmdLine( [ '-i', 'x' ] )

    assert args.backend == 'serial'

#----------------------------------------------------------------------
def test_default_kernel_on():
    '''Kernel is on by default ; --pyedm opts out.'''

    a = gmn.CLI_Parser.ParseCmdLine( [ '-i', 'x' ] )
    assert a.pyedm is False

    b = gmn.CLI_Parser.ParseCmdLine( [ '-i', 'x', '--pyedm' ] )
    assert b.pyedm is True

#----------------------------------------------------------------------
def test_cli_run_app( tmp_path ):
    '''End-to-end : Run.py drives the ABCD config on the kernel default
       path via its real __main__ entry, writing to a temp dir.

    Uses config/default-noPlot.cfg ( showPlot False ) so the run is
    non-interactive. Output goes to an explicit .csv via -o. Asserts a
    clean exit and a well-formed output CSV. Correctness, not timing.
    '''

    repoRoot = os.path.dirname( os.path.dirname( os.path.abspath(
                                __file__ ) ) )
    appPath  = os.path.join( repoRoot, 'apps', 'Run.py' )
    cfgDir   = os.path.join( repoRoot, 'config' )
    outCsv   = os.path.join( str( tmp_path ), 'cli_out.csv' )

    # -o must be a full .csv path ( Run.py rejects a bare directory ).
    cmd = [ sys.executable, appPath, '-i', 'default-noPlot.cfg',
            '-o', outCsv ]

    # Run from config/ so the config's relative network / data paths
    # resolve.
    r = subprocess.run( cmd, capture_output = True, text = True,
                        cwd = cfgDir )

    assert r.returncode == 0, f'Run.py failed :\n{r.stderr[-800:]}'
    assert os.path.exists( outCsv ), 'Run.py produced no output CSV'

    df = pd.read_csv( outCsv )
    assert df.shape[ 0 ] > 0 and df.shape[ 1 ] > 1, 'empty / malformed out'
