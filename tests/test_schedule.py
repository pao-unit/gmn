# Tests for gmn.Schedule : flat chunked dispatch. Pure Python ( fake node
# graph, no pyEDM / network data ), so these run everywhere and need no
# fixtures. Covers order-invariance, parallel==serial, and chunk integrity.

import random

import networkx as nx

from gmn.Schedule import ( PoolWidth, ChunkCount, ChunkNodes,
                           MakeWorkerPool, RunStep )

#----------------------------------------------------------------------
def build_fake_graph( n ):
    '''DiGraph of n nodes, each a fake node reading only lastDataOut.'''

    class FakeNode:
        def __init__( self, name ):
            self.name = name
        def Generate( self, lastDataOut ):
            # Deterministic, order-independent : depends only on t-1.
            base = 0 if lastDataOut is None else lastDataOut.get( self.name, 0 )
            return base + hash( self.name ) % 1000

    G     = nx.DiGraph()
    names = [ f'n{i}' for i in range( n ) ]
    G.add_nodes_from( names )

    for i in range( n - 1 ):
        G.add_edge( names[ i ], names[ i + 1 ] )   # real DAG edges

    for name in names:
        G.nodes[ name ]['Node'] = FakeNode( name )

    return G, names

#----------------------------------------------------------------------
def run_horizon( pool, graph, chunks, steps ):
    '''Drive several timesteps ; return the final { name : val } dict.'''

    last = None

    for _ in range( steps ):
        results = RunStep( pool, graph, chunks, last )
        last    = dict( results )

    return last

#----------------------------------------------------------------------
def test_chunk_nodes_partition():
    '''ChunkNodes covers every node exactly once, no loss or duplication.'''

    names = [ f'n{i}' for i in range( 97 ) ]   # prime : awkward remainder

    for cc in [ 1, 3, 8, 40, 97 ]:
        chunks = ChunkNodes( names, cc )
        flat   = [ x for c in chunks for x in c ]
        assert flat == names, ( cc, 'partition altered order or content' )

#----------------------------------------------------------------------
def test_parallel_equals_serial():
    '''Parallel output equals serial output across several chunk counts.'''

    G, names = build_fake_graph( 500 )
    cores    = 8

    serialChunks = ChunkNodes( names, ChunkCount( len( names ), cores ) )
    serial       = run_horizon( None, G, serialChunks, steps = 6 )

    pool = MakeWorkerPool( PoolWidth( len( names ), cores ) )

    for cc in [ 1, ChunkCount( len( names ), cores ), 200 ]:
        chunks = ChunkNodes( names, cc )
        par    = run_horizon( pool, G, chunks, steps = 6 )
        assert par == serial, f'parallel != serial at chunkCount={cc}'

    pool.shutdown()

#----------------------------------------------------------------------
def test_order_invariance():
    '''Shuffling dispatch order leaves per-step output identical.'''

    G, names = build_fake_graph( 300 )

    base = run_horizon( None, G, ChunkNodes( names, 32 ), steps = 5 )

    shuffled = names[ : ]
    random.Random( 0 ).shuffle( shuffled )
    shuf = run_horizon( None, G, ChunkNodes( shuffled, 32 ), steps = 5 )

    assert shuf == base, 'output depends on execution order ( invariant '\
                         'broken )'
