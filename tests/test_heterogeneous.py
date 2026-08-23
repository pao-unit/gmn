#! /usr/bin/env python3
# Validate the kernel honors per-node E / tau / Tp ( GMN's per-node config
# flexibility ) : nodes with DIFFERENT embedding parameters run together in
# one Generate() loop and each must match its own pyEDM oracle. Guards the
# claim that the kernel does NOT impose a static embedding. Run from repo
# root with pyEDM 2.5.7.




import numpy  as np
import pandas as pd
import pyEDM
import pytest

from gmn.KernelData    import NodeLibrary
from gmn.SimplexKernel import Neighbors, Project

#----------------------------------------------------------------------
def check_config( df, libEnd, E, tau, Tp ):
    '''One ( E, tau, Tp ) : kernel neighbors + projection vs pyEDM.'''

    N = df.shape[0]

    S = pyEDM.Simplex( dataFrame = df, columns = 'A B C', target = 'A',
                       lib = f'1 {libEnd}', pred = f'{N-1} {N}',
                       E = E, tau = tau, Tp = Tp, returnObject = True )

    oNbr = np.asarray( S.knn_neighbors )[ -1 ]
    oPrj = float( np.asarray( S.projection )[ -1 ] )

    raw = df[ [ 'A','B','C' ] ].to_numpy()
    tgt = df[ 'A' ].to_numpy()
    lib = NodeLibrary( raw, tgt, E, tau, libEnd, Tp )

    kNbr, kDist = Neighbors( lib, lib.PredVector(), E + 1 )
    kPrj, _     = Project( kDist, lib.TargetAt( kNbr ) )

    nbrOk = np.array_equal( oNbr, kNbr )
    prjOk = abs( oPrj - kPrj ) / max( abs( oPrj ), 1e-9 ) < 1e-4

    return nbrOk and prjOk

#----------------------------------------------------------------------
@pytest.mark.pyedm
def test_heterogeneous_parameters():
    '''Kernel matches pyEDM across a spread of per-node E / tau / Tp.'''

    rows = 80
    t    = np.arange( 1, rows + 1 )
    df   = pd.DataFrame( { 'Time' : t,
                           'A' : np.sin( 0.10 * t ),
                           'B' : np.cos( 0.08 * t ),
                           'C' : np.sin( 0.05 * t ) } )

    configs = [ ( 3, -1, 1 ), ( 8, -1, 1 ), ( 5, -2, 1 ),
                ( 5, -1, 2 ), ( 10, -2, 3 ), ( 4, -1, 3 ) ]

    for E, tau, Tp in configs :
        assert check_config( df, 60, E, tau, Tp ), \
            f'kernel != pyEDM at E={E} tau={tau} Tp={Tp}'

    print( 'PASS TestHeterogeneousParameters '
           f'( {len(configs)} distinct E/tau/Tp configs )' )

#----------------------------------------------------------------------
