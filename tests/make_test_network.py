#! /usr/bin/env python3
# Generate synthetic GMN networks + data for scaling tests (T4).
#
# Produces, for each requested node count N :
#   <outDir>/net_<N>.pkl   : pickle { 'Graph' : DiGraph, 'Map' : dict }
#                            in the exact format Network.__init__ loads
#   <outDir>/data_<N>.csv  : Time column + one float column per node,
#                            the format ReadDataFrame / Network expect
#
# The graph is a DAG (topologically sortable, as GMN requires). Every
# node draws a few predecessors from earlier nodes so each node has real
# multivariate Simplex inputs; node 0 is the target (sink). The data are
# smooth coupled oscillators so Simplex has a genuine manifold to embed
# rather than noise, making per-node work representative of real runs.
#
# Run from the repo root ( so imports resolve ) :
#   python3 tests/make_test_network.py --sizes 200 500 1000 \
#           --rows 1200 --drivers 5 --seed 0 --out tests/scaling
#
# Then time a size with the RunNoConfig CLI ( see the printed commands ).

import argparse
import os
import pickle

import numpy  as np
import pandas as pd
from   networkx import DiGraph, is_directed_acyclic_graph, topological_sort

#----------------------------------------------------------------------
def BuildGraph( nodeCount, drivers, rng ):
    '''Build a DAG of nodeCount nodes, each with up to `drivers`
       predecessors drawn from lower-indexed nodes ( guarantees acyclic ).

    Arguments:
        nodeCount : int. Number of nodes ( names 'N0'..'N{n-1}' ).
        drivers   : int. Target in-degree per node ( inputs per node ).
        rng       : numpy Generator for reproducible edge choice.

    Returns:
        ( DiGraph, names, driverMap ). driverMap is { node : [drivers] }
        for the pickle 'Map' field.
    '''

    names = [ f'N{i}' for i in range( nodeCount ) ]
    G     = DiGraph()
    G.add_nodes_from( names )

    driverMap = { name : [] for name in names }

    # Node i draws its drivers from nodes 0..i-1 : lower index -> acyclic.
    for i in range( 1, nodeCount ) :
        poolSize = min( i, drivers )
        picks    = rng.choice( i, size = poolSize, replace = False )

        for j in picks :
            src = names[ int( j ) ]
            dst = names[ i ]
            G.add_edge( src, dst )        # driver -> driven
            driverMap[ dst ].append( src )

    # Acyclic by construction; assert to fail loud if edited wrongly.
    assert is_directed_acyclic_graph( G ), 'graph is not a DAG'

    return G, names, driverMap

#----------------------------------------------------------------------
def BuildData( names, rows, rng ):
    '''Coupled-oscillator series : one smooth float column per node.

    Each node is a phase-shifted sinusoid plus light coupling to a
    couple of other nodes and small noise. Smooth dynamics give Simplex
    a real low-dimensional manifold to embed ( representative per-node
    cost ), unlike white noise.

    Arguments:
        names : list[str]. Node names -> column names.
        rows  : int. Number of time rows ( must exceed predictionStart ).
        rng   : numpy Generator.

    Returns:
        pandas.DataFrame with a leading integer 'Time' column.
    '''

    n = len( names )
    t = np.arange( rows )

    # Distinct frequency / phase per node for variety.
    freq  = 0.01 + 0.05 * rng.random( n )
    phase = 2.0  * np.pi * rng.random( n )

    base = np.sin( freq[ None, : ] * t[ :, None ] + phase[ None, : ] )

    # Light coupling : each node adds a fraction of two others' signals.
    coupled = base.copy()

    for k in range( n ) :
        a = int( rng.integers( n ) )
        b = int( rng.integers( n ) )
        coupled[ :, k ] += 0.15 * base[ :, a ] + 0.10 * base[ :, b ]

    # Small observational noise so neighbors are not degenerate.
    coupled += 0.01 * rng.standard_normal( ( rows, n ) )

    data = pd.DataFrame( coupled, columns = names )
    data.insert( 0, 'Time', np.arange( 1, rows + 1 ) ) # 1-based Time

    return data

#----------------------------------------------------------------------
def WriteOne( nodeCount, rows, drivers, seed, outDir ):
    '''Generate and write one ( network pickle, data csv ) pair.'''

    rng = np.random.default_rng( seed )

    G, names, driverMap = BuildGraph( nodeCount, drivers, rng )
    data                = BuildData( names, rows, rng )

    # Pickle format is exactly CreateNetwork's : { 'Graph', 'Map' }.
    network = { 'Graph' : G, 'Map' : driverMap }

    netPath  = os.path.join( outDir, f'net_{nodeCount}.pkl'  )
    dataPath = os.path.join( outDir, f'data_{nodeCount}.csv' )

    with open( netPath, 'wb' ) as f :
        pickle.dump( network, f )
    data.to_csv( dataPath, index = False )

    # Target node = the topological sink ( last in the ordering ).
    target = list( topological_sort( G ) )[ -1 ]

    print( f'  wrote {netPath} ( {G.number_of_nodes()} nodes, '
           f'{G.number_of_edges()} edges )' )
    print( f'  wrote {dataPath} ( {data.shape[0]} rows x '
           f'{data.shape[1]} cols )' )

    return netPath, dataPath, target

#----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser( description = 'Make GMN test nets' )

    parser.add_argument( '--sizes', nargs = '+', type = int,
                         default = [ 200, 500, 1000 ],
                         help = 'Node counts to generate.' )
    parser.add_argument( '--rows', type = int, default = 1200,
                         help = 'Time rows ( must exceed predictionStart ).' )
    parser.add_argument( '--drivers', type = int, default = 5,
                         help = 'Inputs (predecessors) per node.' )
    parser.add_argument( '--seed', type = int, default = 0,
                         help = 'Base RNG seed.' )
    parser.add_argument( '--out', type = str, default = 'tests/scaling',
                         help = 'Output directory.' )
    parser.add_argument( '--predictionStart', type = int, default = 1000,
                         help = 'Suggested predictionStart for run cmd.' )
    parser.add_argument( '--predictionLength', type = int, default = 100,
                         help = 'Suggested predictionLength for run cmd.' )

    args = parser.parse_args()

    os.makedirs( args.out, exist_ok = True )

    print( f'Generating networks {args.sizes} into {args.out}/' )

    for i, n in enumerate( args.sizes ) :
        print( f'[{n} nodes]' )
        netPath, dataPath, target = WriteOne( n, args.rows, args.drivers,
                                              args.seed + i, args.out )

        # Print a ready-to-run RunNoConfig timing command per size.
        print( '  time it with:' )
        print( f'    python apps/RunNoConfig.py -md generate '
               f'-tn {target} \\' )
        print( f'      -nf {netPath} -nd {dataPath} \\' )
        print( f'      -pS {args.predictionStart} '
               f'-pL {args.predictionLength} \\' )
        print( f'      -fn Simplex -E 5 -tau -1 -Tp 1 '
               f'-c 20 -b parallel' )
        print()

    print( 'To compare serial vs parallel, run the same command with '
           '-b serial and time both.' )

#----------------------------------------------------------------------
if __name__ == '__main__':
    main()
