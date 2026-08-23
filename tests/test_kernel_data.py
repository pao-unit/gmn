#! /usr/bin/env python3
# T8 tests : embedding parity + frozen-library / bounded-memory invariant.
# Requires pyEDM 2.5.7 as the embedding oracle. Run from repo root.




import numpy  as np
import pandas as pd
import pyEDM
import pytest

from gmn.KernelData import EmbedColumns, NodeLibrary

#----------------------------------------------------------------------
def Frame( rows = 60 ):
    t = np.arange( 1, rows + 1 )
    return pd.DataFrame( { 'Time' : t,
                           'A' : np.sin( 0.10 * t ),
                           'B' : np.cos( 0.08 * t ),
                           'C' : np.sin( 0.05 * t ) } )

#----------------------------------------------------------------------
@pytest.mark.pyedm
def test_embedding_parity():
    '''EmbedColumns matches pyEDM Embed() across E and tau.'''

    df = Frame()

    for E in ( 2, 3, 5 ):
        for tau in ( -1, -2 ):
            ref  = pyEDM.Embed( dataFrame = df, E = E, tau = tau,
                                columns = 'A B C' ).to_numpy()
            mine = EmbedColumns( df[ [ 'A','B','C' ] ].to_numpy(), E, tau )

            assert np.array_equal( np.isnan( ref ), np.isnan( mine ) ), \
                f'nan mismatch E={E} tau={tau}'

            fin = ~np.isnan( ref ) & ~np.isnan( mine )
            if fin.any() :
                md = np.abs( ref[ fin ] - mine[ fin ] ).max()
                assert md < 1e-5, f'value mismatch E={E} tau={tau} : {md}'

    print( 'PASS TestEmbeddingParity' )

#----------------------------------------------------------------------
@pytest.mark.pyedm
def test_frozen_library_and_memory():
    '''Library matches oracle, PredVector correct, library never grows.'''

    df  = Frame()
    E, tau, Tp, libEnd = 3, -1, 1, 50
    raw = df[ [ 'A','B','C' ] ].to_numpy()
    tgt = df[ 'A' ].to_numpy()

    lib = NodeLibrary( raw, tgt, E, tau, libEnd, Tp )

    # Frozen library matches pyEDM embedding of the same rows. The
    # library spans [ span, libEnd - Tp ) : leading nan rows and the
    # trailing Tp rows ( whose target+Tp exits the library ) are dropped.
    ref  = pyEDM.Embed( dataFrame = df, E = E, tau = tau,
                        columns = 'A B C' ).to_numpy()
    span = ( E - 1 ) * abs( tau )
    refLib = ref[ np.arange( span, libEnd - Tp ) ].astype( np.float32 )
    assert np.abs( refLib - lib.libEmbedded ).max() < 1e-5, 'library mismatch'

    # PredVector matches the last embedded row.
    pv = lib.PredVector()
    assert np.abs( pv - ref[ -1 ].astype( np.float32 ) ).max() < 1e-5, \
        'PredVector mismatch'

    # Memory invariant : Append never changes the library.
    shape0 = lib.libEmbedded.shape
    sum0   = float( lib.libEmbedded.sum() )

    for step in range( 25 ):
        lib.Append( raw[ -1 ] + 0.001 * step )

    assert lib.libEmbedded.shape == shape0, 'library grew'
    assert float( lib.libEmbedded.sum() ) == sum0, 'library content changed'
    assert lib.rollBuffer.shape[0] == span + 1, 'rolling buffer wrong size'

    print( 'PASS TestFrozenLibraryAndMemory' )

#----------------------------------------------------------------------
