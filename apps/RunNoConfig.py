#! /usr/bin/env python3

# Python distribution
from os  import environ
from argparse import ArgumentParser

# Community
import gmn
import gmn.Parameters as Parameters

# Local

# Silence: Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome
environ["XDG_SESSION_TYPE"] = "xcb"

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def main():
    '''Run GMN without a config file. Parameters set by command line args.
       Node specific parameters cannot be set. 
    '''

    args = ParseCmdLine()
    args.configDir  = "NA"
    args.configFile = "NA"

    if args.verbose :
        print( args )

    P = Parameters()
    P.mode            = args.mode
    P.predictionStart = args.predictionStart
    P.predictionLength= args.predictionLength
    P.outPath         = args.outPath
    P.dataOutFile     = args.dataOutFile
    P.showPlot        = args.showPlot
    P.plotType        = args.plotType
    P.plotColumns     = args.plotColumns
    P.plotFile        = args.plotFile
    P.networkName     = args.networkName
    P.targetNode      = args.targetNode
    P.networkFile     = args.networkFile
    P.networkData     = args.networkData
    P.function        = args.function
    P.lib             = args.lib
    P.pred            = args.pred
    P.E               = args.E
    P.Tp              = args.Tp
    P.knn             = args.knn
    P.tau             = args.tau
    P.exclusionRadius = args.exclusionRadius
    P.validLib        = args.validLib
    P.factor          = args.factor
    P.offset          = args.offset

    # Node specific data only with config file for node
    P.nodeInfo        = None
    P.nodeData        = None
    P.columns         = ''
    P.target          = ''
    P.embedded        = False
    P.generateSteps   = 0

    G = gmn.GMN( args, P )
    if "generate" in P.mode.lower() :
        G.Generate() # Run GMN forward in time
    else :
        G.Forecast() # lib & pred forecast : no generation

#----------------------------------------------------------------------------
def ParseCmdLine( argv = None ):
    '''CLI argument parser for most config arguments and parameters'''

    parser = ArgumentParser( description = 'GMN Run No Config' )

    #-------------------------------------------------------------
    # Following are GMN arguments
    #-------------------------------------------------------------
    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output file.')

    parser.add_argument('-r', '--round',
                        dest    = 'round', type = int, 
                        action  = 'store',
                        default = 6,
                        help    = 'DataOut output file precision.')

    parser.add_argument('-c', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-b', '--backend',
                        dest    = 'backend', type = str,
                        action  = 'store',
                        default = 'serial',
                        choices = [ 'parallel', 'serial' ],
                        help    = 'Node dispatch: serial (default). '
                                  'parallel flat thread pool is available '
                                  'but not beneficial for the kernel at '
                                  'typical per-node costs.')

    parser.add_argument('-ch', '--chunks',
                        dest    = 'chunks', type = int,
                        action  = 'store',
                        default = None,
                        help    = 'Override dispatch chunk count '
                                  '(default 4 x cores).')

    parser.add_argument('-kw', '--kdworkers',
                        dest    = 'kdWorkers', type = int,
                        action  = 'store',
                        default = None,
                        help    = 'Override scipy KDTree.query workers per '
                                  'node (default: serial -1, parallel 1).')

    parser.add_argument('-K', '--pyedm',
                        dest    = 'pyedm',
                        action  = 'store_true',
                        default = False,
                        help    = 'Force the pyEDM reference path for all '
                                  'nodes. Default is the float32 kernel for '
                                  'eligible default-path nodes ( pyEDM is '
                                  'used automatically where required ).')

    parser.add_argument('-t', '--threads',
                        dest    = 'threads', type = int,
                        action  = 'store',
                        default = 2,
                        help    = 'OpenMP threads (kedm).')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot.')

    parser.add_argument('-S', '--StatePlot',
                        dest   = 'StatePlot',
                        action = 'store_true', default = False,
                        help   = 'State-space Plot.')

    parser.add_argument('-C', '--PlotColumns', nargs = '+',
                        dest    = 'plotColumns', type = str, 
                        action  = 'store',
                        default = [],
                        help    = 'Plot columns.')

    parser.add_argument('-fs', '--FigureSize', nargs = 2,
                        dest    = 'FigureSize', type = float,
                        action  = 'store',
                        default = [8,8], # inches
                        help    = 'Figure Size.')

    parser.add_argument('-F', '--PlotFile',
                        dest   = 'PlotFile',
                        action = 'store', default = None,
                        help   = 'Write plot to PlotFile.')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose', # type = bool, 
                        action = 'store_true', default = False )

    parser.add_argument('-D', '--DEBUG',
                        dest   = 'DEBUG', # type = bool, 
                        action = 'store_true', default = False )

    #-------------------------------------------------------------
    # Following are Parameters normally specified in a config file
    #-------------------------------------------------------------
    parser.add_argument('-md', '--mode',
                        dest    = 'mode',  type = str, 
                        action  = 'store', default = 'generate',
                        help    = 'GMN mode: generate or forecast')

    parser.add_argument('-pS', '--predictionStart',
                        dest    = 'predictionStart', type = int, 
                        action  = 'store',           default = 0,
                        help    = 'predictionStart of generative mode.')

    parser.add_argument('-pL', '--predictionLength',
                        dest    = 'predictionLength', type = int, 
                        action  = 'store',            default = 10,
                        help    = 'number of steps to generate')

    parser.add_argument('-op', '--outPath',
                        dest    = 'outPath', type = str, 
                        action  = 'store',   default = './',
                        help    = 'output file path')

    parser.add_argument('-do', '--dataOutFile',
                        dest    = 'dataOutFile', type = str, 
                        action  = 'store',      default = '',
                        help    = 'DataOut file')

    parser.add_argument('-Ps', '--showPlot',
                        dest    = 'showPlot', 
                        action  = 'store_true', default = False,
                        help    = 'show plot')

    parser.add_argument('-PT', '--plotType',
                        dest    = 'plotType', type = str, 
                        action  = 'store',    default = 'state',
                        help    = 'plot type: time or state')

    parser.add_argument('-PC', '--plotColumns', nargs = '*',
                        dest    = 'plotColumns', type = str, 
                        action  = 'store', default = [],
                        help    = 'Columns to plot')

    parser.add_argument('-PF', '--plotFile',
                        dest    = 'plotFile', type = str, 
                        action  = 'store', default = '',
                        help    = 'plot file name')

    parser.add_argument('-nn', '--networkName',
                        dest    = 'networkName', type = str, 
                        action  = 'store',       default = '',
                        help    = 'networkName')

    parser.add_argument('-tn', '--targetNode', required = True,
                        dest    = 'targetNode', type = str, 
                        action  = 'store',      default = '',
                        help    = 'target Node')

    parser.add_argument('-nf', '--networkFile', required = True,
                        dest    = 'networkFile', type = str, 
                        action  = 'store', default = '',
                        help    = 'network File')

    parser.add_argument('-nd', '--networkData', required = True,
                        dest    = 'networkData', type = str, 
                        action  = 'store', default = '',
                        help    = 'network Data file')

    parser.add_argument('-fn', '--function',
                        dest    = 'function', type = str, 
                        action  = 'store',    default = 'Simplex',
                        help    = 'node function')

    parser.add_argument('-l', '--lib',
                        dest    = 'lib',      type = str, 
                        action  = 'store', default = '',
                        help    = 'lib')

    parser.add_argument('-p', '--pred',
                        dest    = 'pred',      type = str, 
                        action  = 'store', default = '',
                        help    = 'pred')

    parser.add_argument('-E', '--E',
                        dest    = 'E',     type = int, 
                        action  = 'store', default = 3,
                        help    = 'time delay embedding dimension.')

    parser.add_argument('-Tp', '--Tp',
                        dest    = 'Tp',    type = int, 
                        action  = 'store', default = 1,
                        help    = 'Prediction interval')

    parser.add_argument('-k', '--knn',
                        dest    = 'knn',   type = int,
                        action  = 'store', default = 0,
                        help    = 'knn')

    parser.add_argument('-tau', '--tau',
                        dest    = 'tau',   type = int, 
                        action  = 'store', default = -1,
                        help    = 'tau')

    parser.add_argument('-xr', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int, 
                        action  = 'store',           default = 0,
                        help    = 'exclusion Radius')

    parser.add_argument('-vL', '--validLib', nargs = '*',
                        dest    = 'validLib', type = int, 
                        action  = 'store',    default = [],
                        help    = 'validLib')

    parser.add_argument('-fc', '--factor',
                        dest    = 'factor', type = float, 
                        action  = 'store',  default = 1.0,
                        help    = 'factor')

    parser.add_argument('-os', '--offset',
                        dest    = 'offset', type = float, 
                        action  = 'store',  default = 0.,
                        help    = 'offset')

    args = parser.parse_args( argv ) # if argv is None : default = sys.argv[1:]

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
