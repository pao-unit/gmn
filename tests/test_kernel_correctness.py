# Kernel correctness against a FROZEN pyEDM fixture. Always-on : no live
# pyEDM, only numpy + the kernel + a checked-in reference CSV that was
# generated once by the pyEDM path ( see tests/fixtures/ ). This is the
# routine kernel gate ; live pyEDM parity is the separate -m pyedm suite.

import os

import numpy  as np
import pandas as pd

import gmn

# Frozen pyEDM reference : ABCD network, 100 generated points, produced by
# the --pyedm path from config/default-noPlot.cfg.
FIXTURE   = 'abcd_pyedm_generate_100.csv'
# Generative feedback compounds per-step float32 differences ( ~1e-6 ) over
# the horizon, so this is a trajectory divergence band, not a per-step
# tolerance. 1e-3 over 100 steps ; same-basin agreement ( see Family C ).
REL_TOL   = 1e-3

#----------------------------------------------------------------------
def _fixture_path():
    '''Absolute path to the checked-in fixture directory.'''

    here = os.path.dirname( os.path.abspath( __file__ ) )
    return os.path.join( here, 'fixtures', FIXTURE )

#----------------------------------------------------------------------
def test_kernel_matches_pyedm_fixture( in_config_dir ):
    '''Default ( kernel ) DataOut agrees with the frozen pyEDM fixture.

    Runs the ABCD network at 100 points through the kernel default path
    and compares to the checked-in pyEDM reference. Fast : no live pyEDM.
    '''

    ref = pd.read_csv( _fixture_path() )

    args            = gmn.CLI_Parser.ParseCmdLine( [] )
    args.configFile = 'default-noPlot.cfg'
    args.configDir  = None
    parameters      = gmn.ConfigParser.ReadConfig( args )
    parameters.predictionLength = 100

    G = gmn.GMN( args, parameters )    # kernel on by default
    G.Generate()

    a = ref.select_dtypes( 'number' ).to_numpy()
    b = G.DataOut.select_dtypes( 'number' ).to_numpy()

    assert a.shape == b.shape, f'shape {a.shape} vs {b.shape}'

    fin = np.isfinite( a ) & np.isfinite( b )
    rel = np.abs( a[ fin ] - b[ fin ] ) / np.maximum( np.abs( a[ fin ] ),
                                                       1e-9 )

    assert rel.max() < REL_TOL, f'kernel vs fixture max rel {rel.max():.2e}'
