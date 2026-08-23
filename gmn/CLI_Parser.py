# Python distribution modules
from argparse import ArgumentParser

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseCmdLine( argv = None ):

    parser = ArgumentParser( description = 'GMN' )

    parser.add_argument('-i', '--configFile',
                        dest    = 'configFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input config file.')

    parser.add_argument('-d', '--configDir',
                        dest    = 'configDir', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Directory of input config files.')

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

    args = parser.parse_args( argv ) # if argv is None : default = sys.argv[1:]

    return args
