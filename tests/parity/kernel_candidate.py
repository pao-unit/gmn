#! /usr/bin/env python3
# Kernel candidate adapter for the T7 parity runner. Wraps the T8/T10
# kernel ( KernelData.NodeLibrary + SimplexKernel ) in the candidate
# Simplex() interface the runner expects, returning the three layers in
# the frozen-fixture layout. Tier-2 / non-default geometry returns None
# ( defer to pyEDM fallback ), which the runner counts as a guard pass.

import sys, os
sys.path.insert( 0, os.path.join( os.path.dirname( __file__ ), '..', '..',
                                  'gmn' ) )

import numpy as np

from KernelData   import NodeLibrary
from SimplexKernel import Neighbors, Project, KernelEligible

#----------------------------------------------------------------------
def _ParseRange( spec ):
    '''"a b" -> ( int a, int b ) using pyEDM's 1-based inclusive form.'''
    parts = spec.split()
    return int( parts[0] ), int( parts[1] )

#----------------------------------------------------------------------
class KernelCandidate:
    '''Adapts the float32 kernel to the parity-runner candidate API.'''

    def Simplex( self, inputValues, columns, cols, target, lib, pred,
                 E, tau, Tp, knn, exclusionRadius, validLib ):
        '''Return kernel layers for the LAST pred row, or None to defer.

        Only the default generative geometry is handled ; anything else
        ( exclusion, validLib, straddling pred inside library ) returns
        None so the runner records a tier-2 deferral.
        '''

        # Defer immediately on any non-default configuration.
        if ( exclusionRadius and exclusionRadius > 0 ) or \
           ( validLib is not None and len( validLib ) > 0 ) :
            return None

        colNames = list( columns )
        useCols  = cols.split()
        colIdx   = [ colNames.index( c ) for c in useCols ]
        tgtIdx   = colNames.index( target )

        values = np.asarray( inputValues, float )
        raw    = values[ :, colIdx ]
        tgt    = values[ :, tgtIdx ]

        libStart, libEnd = _ParseRange( lib )   # 1-based inclusive
        predStart, predEnd = _ParseRange( pred )

        # Default path requires pred beyond the library ( libOverlap
        # False ). A straddling / overlapping pred defers.
        if predStart <= libEnd :
            return None

        node = NodeLibrary( raw, tgt, E, tau, libEnd_i = libEnd, Tp = Tp )

        if not KernelEligible( node, exclusionRadius, validLib, False ) :
            return None

        k       = knn if knn > 0 else ( E + 1 )
        predVec = node.PredVector()
        nbrRows, nbrDist = Neighbors( node, predVec, k )
        projection, weights = Project( nbrDist, node.TargetAt( nbrRows ) )

        # Return single-row layers matching fixture layout ( last pred ).
        return {
            'embedding'     : node.libEmbedded,      # library embedding
            'knn_neighbors' : nbrRows.reshape( 1, -1 ),
            'knn_distances' : nbrDist.reshape( 1, -1 ),
            'weights'       : weights.reshape( 1, -1 ),
            'projection'    : np.array( [ projection ], float ),
        }
