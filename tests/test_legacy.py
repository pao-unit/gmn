# Single legacy-correctness gate for the pyEDM reference path. The kernel
# is the default ; this one test forces --pyedm to prove the reference
# pipeline still produces the frozen output. Marked 'pyedm' ( opt-in, and
# skipped without pyEDM >= 2.5.7 ). ABCD network, 100 points.

import os

import pandas as pd
import pytest

import gmn

FIXTURE = 'abcd_pyedm_generate_100.csv'

#----------------------------------------------------------------------
def _fixture_path():
    here = os.path.dirname( os.path.abspath( __file__ ) )
    return os.path.join( here, 'fixtures', FIXTURE )

#----------------------------------------------------------------------
@pytest.mark.pyedm
def test_pyedm_reference_generate( in_config_dir ):
    '''--pyedm path reproduces the frozen ABCD reference exactly.

    The kernel is the default ; forcing kernel=False exercises the pyEDM
    reference pipeline and must match the checked-in fixture that was
    generated the same way. Exact comparison at 6-decimal rounding.
    '''

    ref = pd.read_csv( _fixture_path() )

    args            = gmn.CLI_Parser.ParseCmdLine( [] )
    args.configFile = 'default-noPlot.cfg'
    args.configDir  = None
    args.pyedm      = True
    parameters      = gmn.ConfigParser.ReadConfig( args )
    parameters.predictionLength = 100

    G = gmn.GMN( args, parameters )
    G.Generate()

    assert ref.round( 6 ).equals( G.DataOut.round( 6 ) )
