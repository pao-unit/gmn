# pytest configuration and shared fixtures for the GMN test suite.
#
# Provides:
#   - a pyEDM version gate ( >= 2.5.7, the parity oracle ) so tests that
#     need the reference implementation SKIP with a clear reason rather
#     than error when pyEDM is absent or too old
#   - markers : 'pyedm' ( needs pyEDM >= 2.5.7 ), 'timing' ( host-
#     dependent characterization, never a pass/fail gate )
#   - working-directory handling so config-file tests resolve their
#     relative paths ( the legacy tests read local .cfg / .csv files )
#
# Run the whole suite :   pytest tests/
# Skip timing chars    :   pytest tests/ -m 'not timing'
# Only pyEDM parity    :   pytest tests/ -m pyedm

import os
import pytest

# Minimum pyEDM the parity fixtures were captured against ( 2.5.7,
# commit 4ad6b04 : deterministic tie ordering + nan masking ). Newer is
# allowed ; if a newer pyEDM changes tie / nan behavior the parity tests
# fail loudly rather than skipping.
MIN_PYEDM = ( 2, 5, 7 )

#----------------------------------------------------------------------
def _PyedmVersion():
    '''Return installed pyEDM version tuple, or None if not importable.'''

    try :
        import pyEDM
    except Exception :
        return None

    raw   = getattr( pyEDM, '__version__', '0.0.0' )
    parts = []

    for p in raw.split( '.' )[ :3 ] :
        digits = ''.join( c for c in p if c.isdigit() )
        parts.append( int( digits ) if digits else 0 )

    while len( parts ) < 3 :
        parts.append( 0 )

    return tuple( parts )

#----------------------------------------------------------------------
def pytest_configure( config ):
    '''Register custom markers so `pytest --strict-markers` is clean.'''

    config.addinivalue_line(
        'markers',
        'pyedm: requires pyEDM >= 2.5.7 ( parity oracle ); skipped '
        'otherwise.' )
    config.addinivalue_line(
        'markers',
        'timing: host-dependent performance characterization; not a '
        'pass/fail gate.' )

#----------------------------------------------------------------------
@pytest.fixture( scope = 'session' )
def pyedm_ok():
    '''Session fixture : True when pyEDM >= MIN_PYEDM is importable.'''

    ver = _PyedmVersion()

    return ver is not None and ver >= MIN_PYEDM

#----------------------------------------------------------------------
def pytest_collection_modifyitems( config, items ):
    '''Auto-skip @pytest.mark.pyedm tests when the oracle is missing.'''

    ver     = _PyedmVersion()
    haveOra = ver is not None and ver >= MIN_PYEDM

    if haveOra :
        return

    if ver is None :
        reason = 'pyEDM not installed ( parity oracle required )'
    else :
        vstr   = '.'.join( str( x ) for x in ver )
        mstr   = '.'.join( str( x ) for x in MIN_PYEDM )
        reason = f'pyEDM {vstr} < required {mstr} ( parity oracle )'

    skip = pytest.mark.skip( reason = reason )

    for item in items :
        if 'pyedm' in item.keywords :
            item.add_marker( skip )

#----------------------------------------------------------------------
@pytest.fixture
def in_tests_dir():
    '''chdir into the tests/ directory for config-file tests that read
       local .cfg / .csv by relative name, then restore. Keeps those
       tests runnable regardless of the pytest invocation directory.'''

    here = os.path.dirname( os.path.abspath( __file__ ) )
    prev = os.getcwd()
    os.chdir( here )

    yield here

    os.chdir( prev )

#----------------------------------------------------------------------
@pytest.fixture
def in_config_dir():
    '''chdir into config/ so the ABCD config's relative paths
       ( ../network, ../data ) resolve, then restore. Used by the kernel
       correctness gate and the legacy --pyedm gate.'''

    here    = os.path.dirname( os.path.abspath( __file__ ) )
    cfgDir  = os.path.join( os.path.dirname( here ), 'config' )
    prev    = os.getcwd()
    os.chdir( cfgDir )

    yield cfgDir

    os.chdir( prev )
