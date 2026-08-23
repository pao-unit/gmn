
# Python distribution modules
from concurrent.futures import ThreadPoolExecutor, as_completed

# Schedule holds the flat, barrier-free node scheduler and the persistent
# thread pool used by GMN.Generate(). Under the GMN invariant every node at
# timestep t reads only t-1 outputs (GMN.lastDataOut), so all nodes are
# mutually independent within a step : no antichain layering or barriers are
# needed. Nodes are dispatched in coarse chunks to amortize per-task cost;
# the pool's work queue self-balances heterogeneous node run times.
#
# Threads (not processes) are correct for this path : node.Generate() spends
# its time in numpy / pyEDM C code that releases the GIL, and threads share
# memory so node objects and the t-1 row pass by reference with no pickling.
# Heavy-Python plug-in node functions instead route to process sharding
# (the T9 selection rule); that path is not part of this Stage 1 scheduler.

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def PoolWidth( nodeCount, cores ):
    '''Thread-pool width for the generative run.

    Sized to the core budget, not to any topological width : under flat
    dispatch the ready set is the whole node list every timestep. There
    is no gain from more workers than nodes, so tiny networks clamp to
    nodeCount.

    Arguments:
        nodeCount : int >= 1. Number of nodes dispatched per timestep.
        cores     : int >= 1. Compute-thread budget (args.cores).

    Returns:
        int in [1, cores]. min( cores, nodeCount ), floored at 1. A
        return of 1 selects the serial fallback via MakeWorkerPool.
    '''

    # No point in more workers than cores, or more workers than nodes.
    width = min( max( 1, cores ), max( 1, nodeCount ) )

    return width

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def ChunkCount( nodeCount, cores, override = None ):
    '''Number of dispatch chunks per timestep.

    A fixed multiple of the core budget keeps enough chunks above cores
    for the work queue to self-balance heterogeneous node costs, while
    staying far below one-task-per-node dispatch overhead. Clamped to
    nodeCount so small networks are not over-partitioned.

    Arguments:
        nodeCount : int >= 1. Number of nodes to partition.
        cores     : int >= 1. Compute-thread budget (args.cores).
        override  : int or None. Explicit count from --chunks; if given
                    ( > 0 ) it replaces the computed value.

    Returns:
        int in [1, nodeCount]. The number of chunks for ChunkNodes.
    '''

    # Fixed multiplier : a few chunks per core for soft load balancing.
    CHUNKS_PER_CORE = 4

    # Honor an explicit override, else compute from cores.
    if override is not None and override > 0 :
        count = override
    else :
        count = CHUNKS_PER_CORE * max( 1, cores )

    # Never more chunks than nodes; never fewer than one.
    return max( 1, min( count, max( 1, nodeCount ) ) )

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def ChunkNodes( nodeNames, chunkCount ):
    '''Partition the node list into chunkCount contiguous chunks.

    Partitioning is done once at setup and reused every timestep : the
    node set and its order are fixed for the run. Contiguous slices keep
    the partition trivial and deterministic; balancing comes from having
    several chunks per core, not from chunk contents.

    Arguments:
        nodeNames  : list[str]. All node names to dispatch per timestep.
        chunkCount : int >= 1. Typically ChunkCount( ... ).

    Returns:
        list[list[str]]. Contiguous sublists covering nodeNames exactly.
    '''

    # Ceiling division so the last chunk absorbs any remainder.
    total     = len( nodeNames )
    chunkSize = max( 1, ( total + chunkCount - 1 ) // chunkCount )

    # Slice the list into contiguous chunks of chunkSize.
    chunks = []

    for start in range( 0, total, chunkSize ) :
        chunks.append( nodeNames[ start : start + chunkSize ] )

    return chunks

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def MakeWorkerPool( poolWidth ):
    '''Create the persistent thread pool that lives for the whole run.

    ThreadPoolExecutor is correct here : node.Generate() work is numpy /
    pyEDM C code that releases the GIL, and threads share memory so node
    objects and the t-1 row pass by reference with no pickling.

    Arguments:
        poolWidth : int. Typically PoolWidth( nodeCount, cores ).

    Returns:
        ThreadPoolExecutor( max_workers = poolWidth ) when poolWidth > 1,
        else None to select the serial fallback.

    The caller owns the pool and MUST call pool.shutdown(); GMN.Generate
    wraps the time loop in try / finally to guarantee it.
    '''

    # A width of 1 cannot parallelize : signal the serial fallback.
    if poolWidth <= 1 :
        return None

    # Persistent pool : created once, reused across every timestep.
    return ThreadPoolExecutor( max_workers = poolWidth )

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def GenerateChunk( chunk, graph, lastDataOut ):
    '''Worker task : advance every node in one chunk by one timestep.

    Runs in a pool thread, or inline under the serial fallback. Each
    node reads only its own data and the shared, read-only lastDataOut
    (t-1) and mutates no object visible to other nodes, so chunks are
    independent. Values are returned, never committed here.

    Arguments:
        chunk       : list[str]. Node names in this chunk.
        graph       : networkx.DiGraph. Resolves node objects at
                      graph.nodes[ name ]['Node'].
        lastDataOut : pandas.DataFrame single-row t-1 outputs, or None
                      on the first timestep.

    Returns:
        list of ( nodeName, val ). One pair per node in the chunk.
    '''

    # Loop the chunk internally : amortizes per-task dispatch overhead.
    results = []

    for nodeName in chunk :
        node = graph.nodes[ nodeName ]['Node']
        results.append( ( nodeName, node.Generate( lastDataOut ) ) )

    return results

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def RunStep( pool, graph, chunks, lastDataOut ):
    '''Run one timestep : dispatch all chunks and gather results.

    Flat dispatch of the whole node set with a single synchronization
    point at the gather : no per-layer barriers, since all nodes depend
    only on t-1. When pool is None the chunks run inline; results are
    identical (no RNG, disjoint outputs), only serial.

    Performs NO commit : the caller assembles the step's output row from
    the returned pairs, so pandas is never mutated from worker threads.

    Arguments:
        pool        : ThreadPoolExecutor or None. From MakeWorkerPool.
        graph       : networkx.DiGraph. Resolves node objects.
        chunks      : list[list[str]]. From ChunkNodes; fixed for run.
        lastDataOut : pandas.DataFrame (t-1) or None on timestep 0.

    Returns:
        list of ( nodeName, val ). One pair per node across all chunks;
        order is not significant : the caller commits by nodeName.

    Errors: a node exception is re-raised on gather (with its chunk)
    rather than stalling silently.
    '''

    # Serial fallback : no pool, run chunks inline in order.
    if pool is None :
        results = []

        for chunk in chunks :
            results.extend( GenerateChunk( chunk, graph, lastDataOut ) )

        return results

    # Parallel path : submit every chunk to the persistent pool.
    futures = {}

    for chunk in chunks :
        future = pool.submit( GenerateChunk, chunk, graph, lastDataOut )
        futures[ future ] = chunk

    # Gather : extend results as chunks complete. Re-raise a chunk
    # failure with context instead of stalling the timestep.
    results = []

    for future in as_completed( futures ) :
        try :
            results.extend( future.result() )
        except Exception as err :
            chunk = futures[ future ]
            raise RuntimeError( "RunStep(): chunk starting '" +
                                str( chunk[0] ) + "' failed : " +
                                str( err ) )

    return results
