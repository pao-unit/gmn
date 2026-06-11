#! /usr/bin/env python3

# Distribution modules
import argparse, pickle, json
from   datetime import datetime

# Community modules
import matplotlib.pyplot as plt
from   networkx import DiGraph, is_directed_acyclic_graph, draw_networkx
from   networkx import arf_layout, kamada_kawai_layout, circular_layout
from   networkx import shell_layout, spring_layout, spectral_layout
from   networkx import node_link_data, topological_sort
from   pandas   import DataFrame, read_csv, read_feather, read_pickle

# Local modules
from gmn.Auxiliary import ReadDataFrame

#----------------------------------------------------------------------------
def CreateNetwork( interactionMatrix = None, interactionMatrixFile = None,
                   targetCols = [], threshold = 0, numDrivers = 3,
                   maxNodes = None, driversFile = None,
                   driversColumns = ['column','E'], excludeColumns = [],
                   outputFile = None, cmi = False, plotNetwork = False,
                   layout = 'arf', arrowsize = 10, node_size = 30,
                   node_color = '#1f78b4', alpha = 1, width = 1.2,
                   figsize = (8,8), fontSize = 12,
                   verbose = False, debug = False ):

    '''Create a GMN network using networkx directed graph DiGraph.
    An interaction matrix (interactionMatrix DataFrame or file name) is
    required input. targetCols is required input.

    The imatrix rows quantify interaction for each node. The node in the
    first column of a row is the driven node, nodes in the other columns
    along the row are potential drivers. 

    Starting with a list of target nodes (targetCols) link the strongest
    numDrivers to each target node. Then, recursively add nodes that link
    to the established target nodes and drivers.

    numDriversDF is an optional DataFrame with node names in driversColumns[0]
    and node numDrivers in driversColumns[1]. If numDriversDF is None (default)
    all node numDrivers are set to numDrivers, else set default numDrivers
    but replaced with any node mappings specified in numDriversDF.

    Output: Write binary networkx.DiGraph() object to args.outputFile,
    or a .json file if args.outputFile ends with ".json"

    Return { Graph : network_graph, Map : network_dict }.
    '''
    start = datetime.now()

    if verbose :
        print( f"CreateNetwork() {datetime.now()}", flush = True )

    if isinstance( targetCols, str ) :
        targetCols = [ targetCols ] # convert str to []
    if not len( targetCols ) :
        err = 'CreateNetwork() targetCols required'
        raise RuntimeError( err )

    # Read interactionMatrixFile or validate interactionMatrix DataFrame
    iMatrix = GetInteractionMatrix(interactionMatrix     = interactionMatrix,
                                   interactionMatrixFile = interactionMatrixFile,
                                   excludeColumns        = excludeColumns,
                                   verbose               = verbose)

    D_numDrivers = GetNodeDrivers( driversFile    = driversFile,
                                   driversColumns = driversColumns,
                                   numDrivers     = numDrivers,
                                   columns        = iMatrix.columns,
                                   verbose = verbose, debug = debug )

    if verbose :
        print( f'CreateNetwork() Discover... {datetime.now()}', flush = True )

    network_graph = DiGraph() # for assessing graph properties
    network_nodes = []        # nodes already added to network

    # nodes to be explored start at targetCols
    explore_queue = targetCols.copy()  
    network_dict  = {}  # { node : [drivers] }
    network_cycle = {}  # nodes that create loops (not used)
    terminateMaxNodes = False

    # Starting at targetCols, keep adding nodes until no more
    # nodes in explore_queue
    while len( explore_queue ) and not terminateMaxNodes:
        node_id = explore_queue.pop(0)

        if node_id in network_nodes:
            continue

        nodeNumDrivers = D_numDrivers[ node_id ]

        network_nodes.append( node_id )
        network_dict [ node_id ] = [] # empty list of drivers for new node
        network_cycle[ node_id ] = [] # empty list of cyclic nodes for new node

        # Sort & rank iMatrix (ascending if CMI
        # Store node_ids that exceed threshold (or meet if CMI)
        if cmi :
            # Rank iMatrix ascending, not descending
            # df.loc[ x ] returns the row x as a Series.
            top_drivers =\
                iMatrix.loc[ node_id ].sort_values( ascending = True )
            # Threshold iMatrix >= to allow 0 MI as best rank
            top_drivers =\
                top_drivers[( top_drivers >= threshold ) & \
                            ( top_drivers.index != node_id )][:nodeNumDrivers]
        else :
            # df.loc[ x ] returns the row x as a Series.
            top_drivers =\
                iMatrix.loc[ node_id ].sort_values()

            top_drivers =\
                top_drivers[( top_drivers > threshold ) & \
                            ( top_drivers.index != node_id )][:nodeNumDrivers]

        for driver_id in top_drivers.index:
            # Try adding driver -> current node & edge to network
            network_graph.add_node( driver_id )
            network_graph.add_edge( driver_id, node_id )

            if is_directed_acyclic_graph( network_graph ):
                network_dict[ node_id ].append( driver_id )

                if maxNodes is not None:
                    numNodes = network_graph.number_of_nodes()
                    if numNodes >= maxNodes :
                        print(f"Terminating at maxNodes {numNodes}",
                              flush = True)
                        terminateMaxNodes = True
                        break
            else:
                # driver_id created a cycle, remove edge
                network_graph.remove_edge( driver_id, node_id )

                if not network_graph.has_node( node_id ) :
                    # node_id not in network, remove the driver
                    network_graph.remove_node( driver_id )

                # store driver_id in network_cycle (for reporting)
                network_cycle[ node_id ].append( driver_id )

        # Add driver nodes to explore_queue for upstream nodes
        for driver_id in network_dict[ node_id ]:
            explore_queue.append( driver_id )

    print( len( network_graph ), 'nodes', flush = True )
    if cmi :
        print(f'NOTE: --cmi: iMatrix sorted ascending, threshold >= {threshold}',
              flush = True )

    if debug :
        print( "Network Dictionary { Node : ( [Drivers] ), ... }", flush = True )
        print( network_dict, flush = True )
        print( "Network Cycle { Node : [], ... }", flush = True )
        print( network_cycle, flush = True )

    Network = { 'Graph' : network_graph, 'Map' : network_dict }

    print( f'Elapsed time: {datetime.now() - start}', flush = True )

    #----------------------------------------------------------
    if outputFile :
        if verbose :
            print( f'Writing {outputFile} {datetime.now()}', flush = True )

        if ".json" in outputFile[-5:] :
            obj = node_link_data( network_graph, edges = "edges" )
            obj[ 'topological_ordering' ] =\
                list( topological_sort( network_graph ) )
            obj[ 'target_cols' ] = targetCols

            with open( outputFile, 'w' ) as fdOut:
                json.dump( obj, fdOut )
        else:
            with open( outputFile, 'wb' ) as fdOut:
                pickle.dump( Network, fdOut )

        if verbose :
            print( f'Writing complete {datetime.now()}', flush = True )

    #----------------------------------------------------------
    if plotNetwork :
        if   layout == 'arf'     : layout = arf_layout
        elif layout == 'kk'      : layout = kamada_kawai_layout
        elif layout == 'circ'    : layout = circular_layout
        elif layout == 'shell'   : layout = shell_layout
        elif layout == 'spring'  : layout = spring_layout
        elif layout == 'spectal' : layout = spectral_layout
        else                     : layout = spring_layout

        plt.figure( figsize = figsize, tight_layout = True )
        draw_networkx( network_graph,
                       pos = layout( network_graph ),
                       with_labels = True, arrowsize = arrowsize,
                       node_size = node_size, node_color = node_color,
                       alpha = alpha, width = width,
                       font_size = fontSize,
                       font_weight = 'bold', horizontalalignment = 'left',
                       verticalalignment = 'baseline' )
        plt.show()

    return Network

#----------------------------------------------------------------------------
def GetInteractionMatrix( interactionMatrix = None, interactionMatrixFile = None,
                          excludeColumns = [], verbose = False ) :
    if verbose :
        print( f"Read Interaction Matrix {datetime.now()}", flush=True )

    if interactionMatrix is None :
        if '.csv' in interactionMatrixFile[-4:] :
            # Set DataFrame column 0 to index
            iMatrix = read_csv( interactionMatrixFile, index_col = 0 )
        elif '.feather' in interactionMatrixFile[-8:] :
            iMatrix = read_feather( interactionMatrixFile )
        elif '.gz' in interactionMatrixFile[-3:] or \
             '.xz' in interactionMatrixFile[-3:]:
            iMatrix = read_pickle( interactionMatrixFile )
    else :
        iMatrix = interactionMatrix

        if not isinstance( iMatrix, DataFrame ) :
            err = 'GetInteractionMatrix() interactionMatrix must be DataFrame'
            raise RuntimeError( err )

    # Ensure index == columns : NxN matrix
    if not iMatrix.index.equals( iMatrix.columns ) :
        err='GetInteractionMatrix() interactionMatrix is not NxN index==columns'
        raise RuntimeError( err )

    if len( excludeColumns ) :
        iMatrix.drop( index   = excludeColumns, inplace = True, errors='ignore' )
        iMatrix.drop( columns = excludeColumns, inplace = True, errors='ignore' )

    return iMatrix

#----------------------------------------------------------------------------
def GetNodeDrivers( driversFile = None, driversColumns = ['column','E'],
                    numDrivers = 1, columns = None,
                    verbose = False, debug = False ):
    if verbose :
        print( f"Get Node Drivers {datetime.now()}", flush=True )

    # numDrivers_ is a dict mapping { node : numDrivers }
    # Create D_numDrivers initialized with default values of numDrivers
    D_numDrivers = dict( zip( columns, [numDrivers] * len( columns ) ) )

    if not driversFile is None :
        numDriversDF = ReadNodeDrivers( driversFile, verbose, debug )

        # Fill in any node:numDrivers mappings
        # Convert numDriversDF to dict based on driversColumns
        D_numDriversDF = dict( zip(numDriversDF.loc[:,driversColumns[0]],
                                   numDriversDF.loc[:,driversColumns[1]]) )
        # Copy numDrivers from numDriversDF into numDrivers_
        for node in D_numDriversDF.keys() :
            D_numDrivers[ node ] = D_numDriversDF[ node ]

        if verbose :
            print( f'GetNodeDrivers() Assigned {len(numDriversDF)} numDrivers',
                   flush = True )
        if debug :
            print( D_numDrivers, flush = True )

    return D_numDrivers

#----------------------------------------------------------------------------
def ReadNodeDrivers( driversFile = None, verbose = False, debug = False ):
    '''Read data file of node drivers into dict.
       If file extension is .csv .feather .gz .xz : use pandas -> DataFrame
       if file extension is .pkl : use pickle.load
    '''

    if verbose :
        print( f"Read Node Drivers ... {datetime.now()}", flush = True )

    if driversFile is None :
        raise RuntimeError( 'ReadNodeDrivers() driversFile is None' )

    nodeDF = None
    if '.csv' in driversFile[-4:] :
        nodeDF = read_csv( driversFile )
    elif '.feather' in driversFile[-8:] :
        nodeDF = read_feather( driversFile )
    elif '.gz' in driversFile[-3:] or '.xz' in driversFile[-3:]:
        nodeDF = read_pickle( file )
    elif '.pkl' in driversFile[-4:] :
        with open( driversFile, 'rb' ) as f:
            nodeDF = pickle.load( f )
    else :
        err = f'ReadNodeDrivers() unrecognized file type {driversFile}'
        raise RuntimeError( err )

    if verbose :
        print( f"ReadNodeDrivers {len( nodeDF )} node numDrivers", flush = True )
    if debug :
        print( "ReadNodeDrivers nodeDF :", flush = True )
        print( nodeDF, flush = True )

    return nodeDF

#----------------------------------------------------------------------------
def CreateNetwork_CmdLine():
    '''Wrapper for CreateNetwork with command line parsing'''

    args = ParseCmdLine()
    # Call CreateNetwork()
    n = CreateNetwork( None,
                       interactionMatrixFile = args.interactionMatrixFile,
                       targetCols     = args.targetCols,
                       threshold      = args.threshold,
                       numDrivers     = args.numDrivers,
                       maxNodes       = args.maxNodes,
                       driversFile    = args.driversFile,
                       driversColumns = args.driversColumns,
                       excludeColumns = args.excludeColumns,
                       cmi            = args.cmi,
                       outputFile     = args.outputFile,
                       plotNetwork    = args.plotNetwork,
                       layout         = args.layout,
                       arrowsize      = args.arrowsize,
                       node_size      = args.node_size,
                       node_color     = args.node_color,
                       alpha          = args.alpha,
                       width          = args.width ,
                       fontSize       = args.fontSize,
                       verbose        = args.verbose,
                       debug          = args.debug )

#----------------------------------------------------------------------------
def ParseCmdLine():
    parser = argparse.ArgumentParser( description = 'Create Network' )

    parser.add_argument('-i', '--interactionMatrixFile',
                        dest    = 'interactionMatrixFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Interaction matrix file: .csv or .feather.')

    parser.add_argument('-t', '--targetCols', nargs = '+',
                        dest    = 'targetCols', type = str, 
                        action  = 'store',
                        default = [],
                        help    = 'Target columns.')

    parser.add_argument('-T', '--threshold',
                        dest   = 'threshold', type = float,
                        action = 'store',  default = 0,
                        help   = 'threshold of node interaction.')

    parser.add_argument('-d', '--numDrivers',
                        dest   = 'numDrivers', type = int, 
                        action = 'store',  default = 4,
                        help   = 'Number of driver timeseries.')

    parser.add_argument('-m', '--maxNodes',
                        dest   = 'maxNodes', type = int, 
                        action = 'store',  default = None,
                        help   = 'Maximum number of nodes to add.')

    parser.add_argument('-df', '--driversFile',
                        dest   = 'driversFile', type = str, 
                        action = 'store',  default = None,
                        help   = 'File with map of node:numDrivers.')

    parser.add_argument('-dc', '--driversColumns', nargs = 2,
                        dest   = 'driversColumns', type = str, 
                        action = 'store',  default = ['column','E'],
                        help   = 'Column names of node, E in driversFile.')

    parser.add_argument('-x', '--excludeColumns', nargs = '*',
                        dest   = 'excludeColumns', type = str, 
                        action = 'store',  default = [],
                        help   = 'Columns to exclude.')

    parser.add_argument('-c', '--cmi',
                        dest   = 'cmi',
                        action = 'store_true', default = False,
                        help   = 'Interaction matrix row sort ascending,' +\
                                 'threshold >= to allow 0 MI')

    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store', default = None,
                        help    = 'Output file name.')

    parser.add_argument('-P', '--plotNetwork',
                        dest   = 'plotNetwork', 
                        action = 'store_true',  default = False,
                        help   = 'Render network graph.')

    parser.add_argument('-l', '--layout',
                        dest   = 'layout', type = str, 
                        action = 'store',  default = 'spring',
          help = 'Graph plot layout : arf, kk, circ, shell, spring, spectral.')

    parser.add_argument('-as', '--arrowsize',
                        dest   = 'arrowsize', type = int, 
                        action = 'store',  default = 10,
                        help   = 'arrowsize.')

    parser.add_argument('-ns', '--node_size',
                        dest   = 'node_size', type = int, 
                        action = 'store',  default = 30,
                        help   = 'node_size.')

    parser.add_argument('-nc', '--node_color',
                        dest   = 'node_color', type = str, 
                        action = 'store',  default = '#1f78b4',
                        help   = 'node color.')

    parser.add_argument('-al', '--alpha',
                        dest   = 'alpha', type = float, 
                        action = 'store',  default = 1.,
                        help   = 'alpha.')

    parser.add_argument('-lw', '--width',
                        dest   = 'width', type = float, 
                        action = 'store',  default = 1.2,
                        help   = 'edge width.')

    parser.add_argument('-fs', '--fontSize',
                        dest   = 'fontSize', type = int, 
                        action = 'store',  default = 12,
                        help   = 'Node font size.')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose', 
                        action = 'store_true',  default = False,
                        help   = 'Verbose output.')

    parser.add_argument('-g', '--debug',
                        dest   = 'debug', 
                        action = 'store_true',  default = False,
                        help   = 'debug output.')

    args = parser.parse_args()

    if args.verbose :
        print( 'CLI parameters:' )
        print( args )
        print( '', flush = True )

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
#----------------------------------------------------------------------------
if __name__ == "__main__":
    CreateNetwork_CmdLine()
