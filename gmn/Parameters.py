
#---------------------------------------------------------------
#---------------------------------------------------------------
class Parameters:
    '''
    Parameters object for all entities.

    Network has one. Each Node has one.

    If a node.cfg file is found read Parameters and data from it.
    If not found, use Network Parameters and data. See Node.py

    The values assigned here are the defaults for keys omitted from a
    config file : ConfigParser.ReadConfig() overlays only the keys
    present in the file, so a partial config file is valid. Defaults
    are concrete values, not None, since consumers assume the type :
    Plot.plotColumns.split(), Node.target.isspace(), len(dataOutFile).
    '''

    def __init__( self ):
        '''Constructor'''

        # GMN
        self.mode             = 'generate'
        self.predictionStart  = 0
        self.predictionLength = 10
        self.outPath          = './'
        self.dataOutFile      = ''
        self.showPlot         = False
        self.plotType         = 'state'
        self.plotColumns      = ''
        self.plotFile         = ''
        self.backend          = 'serial'
        self.kernel           = True

        # Network
        self.networkName      = ''
        self.targetNode       = ''
        self.networkFile      = ''
        self.networkData      = ''

        # Node
        self.nodeInfo         = ''
        self.function         = 'Simplex'
        self.nodeData         = ''
        self.nodeConfigPath   = ''

        # EDM
        self.lib              = ''
        self.pred             = ''
        self.E                = 3
        self.Tp               = 1
        self.knn              = 0
        self.tau              = -1
        self.theta            = 3.0
        self.exclusionRadius  = 0
        self.columns          = ''
        self.target           = ''
        self.solver           = ''
        self.embedded         = False
        self.validLib         = []
        self.generateSteps    = 0
        self.libSizes         = ''
        self.sample           = 0
        self.random           = False
        self.includeData      = False
        self.seed             = 0

        # Scale
        self.factor           = 1.0
        self.offset           = 0.0

    #-----------------------------------------------------------
    #-----------------------------------------------------------
    def Print( self ):
        print( 'Parameters: ----------------------------------', flush = True )

        # GMN
        print( '\t', 'mode',             self.mode )
        print( '\t', 'predictionStart',  self.predictionStart  )
        print( '\t', 'predictionLength', self.predictionLength )
        print( '\t', 'outPath',          self.outPath     )
        print( '\t', 'dataOutFile',      self.dataOutFile )
        print( '\t', 'showPlot',         self.showPlot    )
        print( '\t', 'plotType',         self.plotType    )
        print( '\t', 'plotColumns',      self.plotColumns )
        print( '\t', 'plotFile',         self.plotFile    )
        print( '\t', 'backend',          self.backend     )
        print( '\t', 'kernel',           self.kernel      )

        # Network
        print( '\t', 'networkName', self.networkName )
        print( '\t', 'targetNode',  self.targetNode  )
        print( '\t', 'networkFile', self.networkFile )
        print( '\t', 'networkData', self.networkData )

        # Node
        print( '\t', 'nodeInfo',       self.nodeInfo )
        print( '\t', 'nodeData',       self.nodeData )
        print( '\t', 'function',       self.function )
        print( '\t', 'nodeConfigPath', self.nodeConfigPath )

        # EDM
        print( '\t', 'lib',             self.lib   )
        print( '\t', 'pred',            self.pred  )
        print( '\t', 'E',               self.E     )
        print( '\t', 'Tp',              self.Tp    )
        print( '\t', 'knn',             self.knn   )
        print( '\t', 'tau',             self.tau   )
        print( '\t', 'theta',           self.theta )
        print( '\t', 'exclusionRadius', self.exclusionRadius )
        print( '\t', 'columns',         self.columns  )
        print( '\t', 'target',          self.target   )
        print( '\t', 'solver',          self.solver   )
        print( '\t', 'embedded',        self.embedded )
        print( '\t', 'validLib',        self.validLib )
        print( '\t', 'generateSteps',   self.generateSteps )
        print( '\t', 'libSizes',        self.libSizes )
        print( '\t', 'sample',          self.sample   )
        print( '\t', 'random',          self.random   )
        print( '\t', 'includeData',     self.includeData )
        print( '\t', 'seed',            self.seed )

        print( '\t', 'factor',          self.factor )
        print( '\t', 'offset',          self.offset, flush = True )
