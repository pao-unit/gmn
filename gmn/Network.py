
# Python distribution modules
import pickle

# Community modules
from networkx import topological_sort

# Local modules 
from .Node      import Node
from .Auxiliary import ReadDataFrame

#---------------------------------------------------------------
#---------------------------------------------------------------
class Network:
    '''
    Network object instantiated from GMN.__init__. Stored in GMN.Network.

    Reads networkx DiGraph file from CreateNetwork.py

    Loads Network data if Parameters.networkData specified. This is default
    data for Nodes. Node will override data if a node configuration file
    is specified (a file in Node.configPath named NodeName.cfg) and the node
    configuration file specifies a Node.data filename.

    Creates Node objects corresponding to GMN.Network.nodes (networkx DiGraph 
    nodes). The Node class objects are stored as "attributes"
    ( dict { 'Node' : Node Object } ) in the corresponding self.Graph.nodes. 
    '''

    def __init__( self, args, parameters ):
        '''Constructor'''
        self.Parameters        = parameters
        self.Graph             = None
        self.NetworkMap        = None
        self.TopologicalSorted = None # Flat node list; also dispatch order
        self.data              = None # All input data subset to dataColumns
        self.dataColumns       = None # time + TopologicalSorted nodes
        self.dataLib_i         = None # indices to subset data "library"
        self.timeColumnName    = None

        # Read network graph : See CreateNetwork.py
        graphFile = parameters.networkFile
        with open( graphFile, 'rb' ) as f :
            NetworkGraphDict = pickle.load( f )
            self.Graph       = NetworkGraphDict[ 'Graph' ]
            self.NetworkMap  = NetworkGraphDict[ 'Map'   ] # Not used

            if args.DEBUG :
                print( '-> Network.__init__()', flush = True )
                self.Parameters.Print()

                print( 'Graph.nodes : ---------------------' )
                print( self.Graph.nodes, flush = True )
                print( 'NetworkMap : ----------------------' )
                print( self.NetworkMap, flush = True )

                if args.Plot :
                    import matplotlib.pyplot as plt
                    from   networkx import draw, shell_layout
                    plt.figure()
                    draw( self.Graph,
                          pos = shell_layout( self.Graph ),
                          node_size = 30, with_labels = True,
                          font_size = 14, font_weight = 'bold', alpha = 0.5 )
                    plt.show()

        # Sort for execution order : target node last
        # Note: topological_sort() returns a generator, store in list for reuse
        self.TopologicalSorted = list( topological_sort( self.Graph ) )

        # INVARIANT (licenses flat, barrier-free dispatch in GMN.Generate):
        # at timestep t every node reads only t-1 outputs (GMN.lastDataOut),
        # never a same-step sibling's output. Nodes are therefore mutually
        # independent within a step, so TopologicalSorted doubles as the
        # flat node list and its order does not affect results. No antichain
        # layering is needed this iteration. See Schedule.RunStep().

        # Load Network data as Pandas DataFrame
        if parameters.networkData and not parameters.networkData.isspace() :
            self.data = ReadDataFrame( parameters.networkData,
                                       verbose = args.verbose )

            if self.data.shape[0] <= parameters.predictionStart - 1:
                errMsg = "Network.__init__(): Nummber of data rows " +\
                         str( self.data.shape[0] ) +\
                         " is less than predictionStart " +\
                         str( parameters.predictionStart )
                raise RuntimeError( errMsg )

            if "generate" in parameters.mode.lower() :
                # Indices for "data library" from index 1 to predictionStart
                self.dataLib_i = range( parameters.predictionStart )
            else :
                # Forecast mode : all data since lib & pred are specified
                self.dataLib_i = range( self.data.shape[0] )

            if args.DEBUG :
                print( "Network.__init__() Loaded",
                       parameters.networkData, " shape :", str(self.data.shape) )
                print( self.data.iloc[ self.dataLib_i ].tail(2), flush = True )

        self.timeColumnName = self.data.columns[0] # PRESUME column 1 is time

        # Subset self.data to network nodes
        self.dataColumns = [self.timeColumnName] + self.TopologicalSorted
        self.data = self.data.loc[:, self.dataColumns]

        # Instantiate Node objects. Store in self.Graph.nodes[ nodeName ]
        #   networkx.org/documentation/stable/reference/classes/digraph.html#
        for nodeName in self.Graph :
            # Node constructor can override Network data
            newNode = Node( args, self, nodeName )

            # Assign newNode object as attribute to this Graph node
            self.Graph.nodes[ nodeName ][ 'Node' ] = newNode

        # Validate all Nodes have data
        for nodeName in self.Graph:
            node = self.Graph.nodes[ nodeName ]['Node']

            if node.data is None:
                errMsg = "Network.__init__(): Node" + node.name +\
                         " has no data."
                raise RuntimeError( errMsg )
