
# Python distribution modules
from datetime import datetime

# Community modules
from pandas import DataFrame, concat
from numpy  import full, nan

# Local modules 
from .Network      import Network
from .Auxiliary    import TimeExtension
from .CLI_Parser   import ParseCmdLine
from .ConfigParser import ReadConfig
from .Schedule     import ( PoolWidth, ChunkCount, ChunkNodes,
                            MakeWorkerPool, RunStep )

#-----------------------------------------------------------------------
class GMN:
    '''Class for Generative Manifold Networks (GMN)
       ../apps/Run.py is a CLI to instantiate, configure and Run GMN.
    '''

    # import Plot as a GMN class method
    from .Plot import Plot

    #-------------------------------------------------------------------
    def __init__( self,
                  args        = None,  parameters = None,
                  configFile  = None,  configDir  = None,
                  outputFile  = None,  cores      = 4,
                  backend     = 'serial', chunks  = None,
                  kdWorkers   = None,  kernel     = True,
                  plot        = False, plotType   = 'state',
                  plotColumns = [],    plotFile   = None,
                  figureSize  = [8,8], verbose    = False, debug = False ):

        '''Constructor

        Full configuration is performed by instantiating GMN with
        args from .CLI_Parser.ParseCmdLine and parameters from
        from .ConfigParser.ReadConfig.

        If args is None GMN.__init__ arguments partially populate args.
        If parameters is None Parameters object is created from the args.
        '''

        if args is None:
            args = ParseCmdLine( argv = [] ) # set default args
            # Insert constructor arguments into args
            args.configFile  = configFile
            args.configDir   = configDir
            args.outputFile  = outputFile
            args.cores       = cores
            args.backend     = backend
            args.chunks      = chunks
            args.kdWorkers   = kdWorkers
            args.kernel      = kernel
            # Keep pyedm consistent with the kernel kwarg so the
            # derivation below ( args.kernel = not args.pyedm ) agrees.
            args.pyedm       = not kernel
            args.Plot        = plot
            args.plotType    = plotType
            args.plotColumns = plotColumns
            args.PlotFile    = plotFile
            args.FigureSize  = figureSize
            args.verbose     = verbose
            args.DEBUG       = debug

        if args.configDir is None and args.configFile is None:
            raise RuntimeError( 'GMN(): configFile is required.' )

        if parameters is None:
            parameters = ReadConfig( args )

        self.args        = args        # command line args
        self.Parameters  = parameters  # args.configFile Parameters
        self.Network     = None
        self.DataOut     = None
        self.lastDataOut = None

        if args.DEBUG :
            import faulthandler
            faulthandler.enable()

        # Resolve the KDTree.query thread budget BEFORE Network builds nodes
        # ( each Node reads args.kdWorkers at construction ). An explicit
        # --kdworkers override wins; otherwise derive from backend so outer
        # and inner parallelism never oversubscribe:
        #   serial   -> -1 : query uses all cores, node loop is serial
        #   parallel ->  1 : outer thread pool supplies concurrency
        override = getattr( args, 'kdWorkers', None )

        if override is not None :
            args.kdWorkers = override
        elif args.backend.lower() == 'serial' :
            args.kdWorkers = -1
        else :
            args.kdWorkers = 1

        # Kernel is ON by default. --pyedm forces the reference path for
        # ALL nodes ( args.pyedm True => kernel False ). When args.pyedm is
        # absent ( programmatic construction ), fall back to the kernel
        # kwarg resolved into args above.
        if hasattr( args, 'pyedm' ) :
            args.kernel = not args.pyedm

        # Instantiate Network : Read DiGraph and instantiate nodes
        self.Network = Network( args, parameters )

        # Kernel setup : freeze each eligible node's float32 library once
        # ( before the time loop ). Ineligible nodes keep KernelLib None
        # and fall back to pyEDM. No-op unless args.kernel is set.
        if getattr( args, 'kernel', False ) :
            nKernel = 0
            for nodeName in self.Network.TopologicalSorted :
                node = self.Network.Graph.nodes[ nodeName ]['Node']
                node.KernelSetup()
                if node.KernelLib is not None :
                    nKernel += 1

            if args.verbose :
                print( f'GMN: kernel enabled for {nKernel} / '
                       f'{len( self.Network.TopologicalSorted )} nodes '
                       f'( rest use pyEDM fallback )' )

        # Allocate DataFrame for output data.
        self.DataOut = DataFrame( columns = self.Network.dataColumns,
                                  dtype = float )

    #-------------------------------------------------------------------
    def Generate( self ):
        '''Execute GMN generative loop for predictionLength steps
           calling the Generate() method of each Network Node. 
        '''

        if self.args.verbose or self.args.DEBUG :
            start = datetime.now()
            print( f'-> GMN:Generate() {start}', flush = True )

        # Local References for convenience and readability
        Network  = self.Network
        Graph    = self.Network.Graph
        nodeList = self.Network.TopologicalSorted # flat node set (order N/A)

        # Column-position map for the vectorized per-step commit : build the
        # step output row as a numpy array indexed here, then wrap it into a
        # DataFrame in one call ( no per-cell pandas writes ).
        dataColumns = self.Network.dataColumns
        colPos      = { name : i for i, name in enumerate( dataColumns ) }

        # Partition the node set once : reused every timestep. --chunks
        # overrides the fixed ( 4 x cores ) default via ChunkCount().
        chunkCount = ChunkCount( len( nodeList ), self.args.cores,
                                 self.args.chunks )
        chunks     = ChunkNodes( nodeList, chunkCount )

        # Select backend : serial (default) ; parallel is opt-in and not
        # beneficial for the kernel at typical per-node costs.
        # PoolWidth() -> 1 (single-node networks) yields a None pool, i.e.
        # the serial fallback, from MakeWorkerPool().
        if self.args.backend.lower() == 'serial' :
            pool = None
        else :
            poolWidth = PoolWidth( len( nodeList ), self.args.cores )
            pool      = MakeWorkerPool( poolWidth )

        # Persistent pool lives for the whole run. try / finally guarantees
        # shutdown() on normal exit and on exception.
        try :
            # Time Loop : strictly serial; step t depends on step t-1.
            for t_i in range( self.Parameters.predictionLength ):
                if self.args.DEBUG :
                    print( "===================== GMN Time:", t_i,
                           "=================================", flush = True )

                # Flat dispatch : all nodes run against the t-1 lastDataOut.
                # Single sync point at the gather; no per-layer barriers.
                results = RunStep( pool, Graph, chunks, self.lastDataOut )

                # Vectorized commit : fill a numpy row by column position,
                # then build the single-row DataFrame in one construction.
                row = full( len( dataColumns ), nan )

                for nodeName, val in results :
                    row[ colPos[ nodeName ] ] = val

                NodeOutput = DataFrame( [ row ], columns = dataColumns )

                # Step output becomes t-1 input; append to DataOut.
                self.lastDataOut = NodeOutput
                self.DataOut     = concat( [ self.DataOut, NodeOutput ] )

        finally :
            # Release worker threads (no-op for the serial fallback).
            if pool is not None :
                pool.shutdown()

        # if factor != 1 apply : factor is a Network-level Parameter
        # ( the prior 'node' loop variable no longer exists here )
        if self.Parameters.factor != 1 :
            self.DataOut = self.DataOut.mul( self.Parameters.factor )

        # Insert time column to DataOut
        # PRESUMED Network data column 1 is time
        newTime = TimeExtension(
            Network.data.iloc[ Network.dataLib_i ][ Network.timeColumnName ],
            self.Parameters.predictionLength )

        self.DataOut[ Network.timeColumnName ] = newTime

        # Reset DataFrame row labels to default 0-offset integers
        self.DataOut.reset_index( drop = True, inplace = True )

        if self.args.verbose or self.args.DEBUG :
            end = datetime.now()
            print( f'<- GMN:Generate() {end}  :  {end-start}', flush = True )

        self.Output()

    #-------------------------------------------------------------------
    def Forecast( self ):
        '''Execute GMN forecast calling the Forecast() method of each 
           Network Node. It is presumed that lib & pred are specified.
        '''

        if self.args.verbose or self.args.DEBUG :
            start = datetime.now()
            print( f'-> GMN:Forecast() {start}', flush = True )

        # Local References for convenience and readability
        Network = self.Network
        Graph   = self.Network.Graph

        # Network Loop
        for nodeName in Network.TopologicalSorted :
            node = Graph.nodes[ nodeName ]['Node']

            if self.args.DEBUG :
                print( "GMN:Forecast Network Loop:", nodeName )
                print( 'columns:', node.Parameters.columns, ':',
                       'target',   node.Parameters.target, flush = True )

            # Call node Forecast method and store in DataOut
            self.DataOut[ nodeName ] = node.Forecast()[1]

            if nodeName == Network.TopologicalSorted[0] :
                # Copy time values : only on first node
                self.DataOut[ Network.timeColumnName ] = node.Forecast()[0]

        # Reset DataFrame row labels to default 0-offset integers
        self.DataOut.reset_index( drop = True, inplace = True )

        if self.args.verbose or self.args.DEBUG :
            end = datetime.now()
            print( f'<- GMN:Forecast() {end}  :  {end-start}', flush = True )

        self.Output()

    #-------------------------------------------------------------------
    def Output( self ):
        '''Write DataOut file(s). Plot'''

        fmt = "%." + str( self.args.round ) + "f"
        
        # Write DataOut file(s)
        if self.args.outputFile:
            if '.csv' in self.args.outputFile[-4:] :
                self.DataOut.to_csv( self.args.outputFile,
                                     float_format = fmt, index = False )
            elif '.feather' in self.args.outputFile[-8:] :
                self.DataOut.to_feather( self.args.outputFile )
            else :
                print( 'GMN.Output(): Unrecognized output file format' )

        if len( self.Parameters.dataOutFile ) and \
           not self.Parameters.dataOutFile.isspace():
            outFile = self.Parameters.outPath + '/' + self.Parameters.dataOutFile
            if '.csv' in outFile[-4:] :
                self.DataOut.to_csv( outFile, float_format = fmt, index = False )
            elif '.feather' in outFile[-8:] :
                self.DataOut.to_feather( outFile )
            else :
                print( 'GMN.Output(): Unrecognized data out file format' )

        if self.args.DEBUG :
            print( "GMN.Output() DataOut:" )
            print( self.DataOut, flush = True )

        # Plot
        if self.args.Plot or self.args.StatePlot or \
           self.Parameters.showPlot or len( self.Parameters.plotFile ):
            self.Plot()
