
# Python distribution modules
from copy    import copy
from pathlib import Path

import numpy as np

# Community modules
from pyEDM import Simplex, SMap, Embed

try:
    from math                 import dist
    from sklearn              import neighbors
    from sklearn.svm          import SVR
    from sklearn.linear_model import LinearRegression
except ImportError as err:
    print( err, "Node: SVR, knn, Linear not available", flush = True )

try:
    from os import environ
    environ[ 'OMP_PROC_BIND' ] = 'false' # Kokkos warning OpenMP
    from kedm import simplex as kedm_simplex
    from kedm import smap    as kedm_smap
except ImportError as err:
    print( err, "Node: kedm not loaded", flush = True )

# Local modules 
from .ConfigParser import ReadConfig
from .Common       import *
from .Auxiliary    import ReadDataFrame

#-----------------------------------------------------------
#-----------------------------------------------------------
class Node:
    '''
    A node in the network. Each node has its own data and Parameters.  

    By default, the node is initialized with a copy() of Network.Parameters
    and Network.data.

    If a Parameter node.cfg file is found: read Parameters and data from it.

    Network output at the end of each timestep is appended to the node data. 
    Thus all nodes share generated data, but can be initialized with 
    individual data and parameters.

    Note the data "library" is limited to Parameters.predictionStart. It 
    does not grow as generated values are added to node data.

    Generate() method performs prediction. Called from GMN Generate() in the
    Network Node loop to dispatch the specified FunctionType and return a 
    single prediction value collected across all nodes in GMN.lastDataOut.
    '''

    # import Generate, Forecast as a Node class method
    from .Generate import Generate
    from .Forecast import Forecast

    #----------------------------------------------------------------------
    def __init__( self, args, Network, nodeName ):
        '''Constructor'''
        self.name         = nodeName
        self.args         = args
        self.Network      = Network
        self.Parameters   = None  # from Network, or node.cfg file, if any
        self.Function     = None  # Reference to Simplex, SMap...
        self.FunctionType = None  # Enumeration
        self.data         = None  # input data (copy or read) + generated
        self.libEnd_i     = None  # EDM library end: Constant @ predictionStart
        # KDTree.query thread budget for pyEDM Simplex, set from backend:
        #   serial   -> -1 (query uses all cores; node loop is serial)
        #   parallel ->  1 (outer thread pool provides concurrency)
        # Default -1 preserves stock pyEDM behavior if unset.
        self.kdWorkers    = getattr( args, 'kdWorkers', -1 )
        self.KernelLib    = None  # float32 kernel library; set by KernelSetup

        if args.DEBUG :
            print( '-> Node.__init__() : ', nodeName, flush = True )

        # Set nodeParameters = True if node config file found
        nodeParameters = False

        if self.Network.Parameters.nodeConfigPath :
            # Use pathlib.Path() class object for OS independence 
            nodePath = Path( self.Network.Parameters.nodeConfigPath )
            nodeFile = Path( self.Network.Parameters.nodeConfigPath + \
                             nodeName + '.cfg' )

            queryGen = nodePath.glob( '*.cfg' ) # Path.glob returns generator
            query    = [ str( x ) for x in queryGen ]

            if str( nodeFile ) in query :
                nodeParameters = True

                if args.DEBUG :
                    print( str( nodeFile ) + " Found in ", query, flush = True )

        if nodeParameters : # Found node.cfg : Get config Parameters
            self.Parameters = ReadConfig( args, nodeFile )
        else : # default to Network config
            self.Parameters = copy( self.Network.Parameters )

        # Assign columns and target : target is the node name
        if len(self.Parameters.target) == 0 or self.Parameters.target.isspace():
             self.Parameters.target = self.name

        # columns are inputs to the node + target
        # target required if node has no input, predict the var itself
        if len(self.Parameters.columns)==0 or self.Parameters.columns.isspace():
            # G.predecessors(n) returns an iterator over predecessor nodes of n.
            self.Parameters.columns = \
                [ n for n in self.Network.Graph.predecessors( self.name ) ]
            self.Parameters.columns.append( self.name )

        # Assign node data
        timeVec      = self.Network.data.columns[0] # PRESUME first col time
        nodeDataCols = [timeVec] + self.Parameters.columns

        if nodeParameters :
            # Found node.cfg : Get data if specified
            if self.Parameters.nodeData :
                data = ReadDataFrame( self.Parameters.nodeData,
                                      verbose = args.verbose )

                if not set( nodeDataCols ).issubset( data.columns ) :
                    msg = "Node(): " + nodeName +\
                        " Node data does not contain " + nodeDataCols
                    raise RuntimeError( msg )

                # Subset to data library, time + target + columns
                # Network: dataLib_i = range( predictionStart )
                self.data = data.loc[ self.Network.dataLib_i,
                                       nodeDataCols ].copy()
            else :
                self.data = Network.data.loc[ self.Network.dataLib_i,
                                              nodeDataCols ].copy()
        else :
            # No Node data specified. Make copy of network data
            # subset to dataLib_i = range( parameters.predictionStart )
            self.data = Network.data.loc[ self.Network.dataLib_i,
                                          nodeDataCols ].copy()

        # Assign node FunctionType and Function
        nodeFunction = self.Parameters.function.lower()

        if "simplex" == nodeFunction :
            self.FunctionType = FunctionType.Simplex
            self.Function     = Simplex

            # EDM lib index based on input data (subset to predictionStart)
            self.libEnd_i = self.data.shape[0]

        elif "smap" == nodeFunction :
            self.FunctionType = FunctionType.SMap
            self.Function     = SMap

            # EDM lib index with offset from embedding time shift
            # since SMap multivariate requires embedded = True
            offset = ( self.Parameters.E ) * abs( self.Parameters.tau )
            self.libEnd_i = self.data.shape[0] - offset

        elif "kedmsimplex" == nodeFunction or "kedm_simplex" == nodeFunction:
            self.FunctionType = FunctionType.kedmSimplex
            self.Function     = kedm_simplex

            # EDM lib index based on input data (subset to predictionStart)
            self.libEnd_i = self.data.shape[0]

        elif "kedmsmap" == nodeFunction or "kedm_smap" == nodeFunction:
            self.FunctionType = FunctionType.kedmSMap
            self.Function     = kedm_smap

            # EDM lib index based on input data (subset to predictionStart)
            self.libEnd_i = self.data.shape[0]

        elif "linear" == nodeFunction :
            self.FunctionType = FunctionType.Linear
            self.Function     = LinearRegression

        elif "svr" == nodeFunction :
            self.FunctionType = FunctionType.SVR
            self.Function     = SVR

        elif "knn" == nodeFunction :
            self.FunctionType = FunctionType.knn
            self.Function     = neighbors.KNeighborsRegressor

        else :
            raise RuntimeError( "Node(): " + nodeName +\
                                " Invalid node function: " + nodeFunction )

        if args.DEBUG :
            print( 'columns:', self.Parameters.columns )
            print( 'target:',  self.Parameters.target  )
            print( str( self.FunctionType ), str( self.Function ) ) 
            print( '<- Node.__init__() : ', nodeName, flush = True )

    #------------------------------------------------------------------
    def KernelSetup( self ):
        '''Build the float32 kernel library for an eligible node.

        Called once before the time loop when args.kernel is set. Only
        default-path Simplex nodes ( no user lib / pred, exclusionRadius
        0, empty validLib ) are eligible ; anything else leaves
        self.KernelLib None so Generate() falls back to pyEDM. Sets:
            self.KernelLib : NodeLibrary or None ( None => use pyEDM )

        The library is frozen here from the node's seed data ( through
        predictionStart ), so the per-step path never re-embeds or
        rebuilds. Memory is O( N_lib x E x nCols ), constant for the run.
        '''

        # Imported here to avoid a hard kernel dependency when unused.
        from .KernelData    import NodeLibrary
        from .SimplexKernel import KernelEligible

        self.KernelLib = None                 # default : pyEDM fallback

        P = self.Parameters

        # Only default-path Simplex is kernel-eligible.
        if self.FunctionType.value != FunctionType.Simplex.value :
            return

        # User-set lib / pred, exclusion, or validLib -> non-default -> defer.
        if len( P.lib ) and not P.lib.isspace() :
            return
        if len( P.pred ) and not P.pred.isspace() :
            return
        if P.exclusionRadius and P.exclusionRadius > 0 :
            return
        if P.validLib is not None and len( P.validLib ) > 0 :
            return

        # Assemble the node's raw input columns ( predecessors + self )
        # and target from the seed data, in the column order Parameters
        # lists, so the embedding matches pyEDM. columns may be a
        # space-separated string ( config ) or a list.
        cols = P.columns
        if isinstance( cols, str ) :
            cols = cols.split()

        target = P.target

        rawValues    = self.data[ cols ].to_numpy()
        targetValues = self.data[ target ].to_numpy()

        lib = NodeLibrary( rawValues, targetValues,
                           E = P.E, tau = P.tau,
                           libEnd_i = self.libEnd_i, Tp = P.Tp )

        # Final eligibility ( library large enough for knn ).
        if KernelEligible( lib, P.exclusionRadius, P.validLib, False ) :
            self.KernelLib = lib

            # Cache integer positions of this node's input columns within
            # the network's dataColumns, so the per-step append can index
            # a plain numpy row by position ( no pandas / Arrow string
            # lookup, which otherwise dominates at scale ).
            dataCols = list( self.Network.dataColumns )
            self.KernelColPos = np.array(
                [ dataCols.index( c ) for c in cols ], dtype = int )
