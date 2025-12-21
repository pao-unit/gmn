# Python distribution modules

# Community modules
import matplotlib.pyplot as plt
from   numpy  import array
from   pandas import to_datetime

# Local modules
from pyEDM import ComputeError

#--------------------------------------------------------------
#--------------------------------------------------------------
def Plot( self ):
    '''Method of class GMN

    Plot either timeseries (time) or time & state-space (state).
    CLI args or parameters GMN.* specify plot options.
    CLI args override config parameters.
    '''

    # Local references for convenience
    args       = self.args
    parameters = self.Parameters
    data       = self.Network.data
    gmnData    = self.DataOut

    plotTitle = parameters.function + "  "       +\
                parameters.mode                  +\
                "  E="   + str( parameters.E   ) +\
                "  tau=" + str( parameters.tau ) +\
                "  Tp="  + str( parameters.Tp  ) 

    if len( args.plotColumns ):
        plotColumns = args.plotColumns # CLI arguments override parameters
    else:
        plotColumns = parameters.plotColumns.split()

    if not len( plotColumns ) :
        raise RuntimeError( "GMN.Plot(): No plot columns found." )

    # Check for invalid columns
    invalidColumns = []
    for plotColumn in plotColumns :
        if plotColumn not in gmnData.columns :
            print( "GMN.Plot():", plotColumn, "not in GMN data, ignoring." )
            invalidColumns.append( plotColumn )

        if plotColumn not in data.columns :
            print( "GMN.Plot():", plotColumn, "not in Network data, ignoring." )
            if plotColumn not in invalidColumns :
                invalidColumns.append( plotColumn )

    if invalidColumns :
        for invalidColumn in invalidColumns:
            plotColumns.remove( invalidColumn )

        print( "GMN.Plot(): new plotColumns:", plotColumns )

    # First column [:, 0] is PRESUMED to be time vector
    timeData = data.iloc   [:, 0]
    timeGMN  = gmnData.iloc[:, 0]

    # if time Series are strings try to convert to pandas Timestamp
    if isinstance( timeData[0], str ) :
        try :
            td = to_datetime( timeData )
            timeData = td
        except :
            pass

    if isinstance( timeGMN[0], str ) :
        try :
            tg = to_datetime( timeGMN )
            timeGMN = tg
        except :
            pass

    # If both time Series are still string, convert to integer sequence
    if isinstance( timeData[0], str ) and isinstance( timeGMN[0],  str ):
        # convert to sequental integer
        timeData = array( [ i for i in range( data.shape[0] ) ] )
        timeGMN  = array( [ i for i in range( gmnData.shape[0] ) ] )

    if "generate" in parameters.mode.lower() :
        timeLib = timeData [ range( parameters.predictionStart ) ]
        dataLib = data.iloc[ range( parameters.predictionStart ), : ]

        # range for observed data over library [ 0 : predictionStart ]
        dataPred_i = range( parameters.predictionStart - 1, data.shape[0] )
        # range for GMN predicted in the observed data
        pred_i = range(parameters.predictionStart,
                       parameters.predictionStart + parameters.predictionLength)
    else :
        # Forecast mode
        library_i  = range( 0, data.shape[0] - gmnData.shape[0] + 1 )
        timeLib    = timeData [ library_i ]
        dataLib    = data.iloc[ library_i, : ]

        dataPred_i = range( data.shape[0] - gmnData.shape[0] - 1, data.shape[0] )
        pred_i     = range( data.shape[0] - gmnData.shape[0], data.shape[0] )

    #----------------------------------------------------------------------
    # Time series plots
    #----------------------------------------------------------------------
    if args.Plot or 'time' in parameters.plotType.lower() :
        # plotColumns rows x 2 columns
        fig, ax = plt.subplots( len( plotColumns ), 2 )
        fig.set_size_inches( args.FigureSize )
        fig.suptitle( plotTitle )
        fig.tight_layout( rect = ( 0, 0, 1, 0.98 ) )

        i = 0
        for col in plotColumns:
            if len( plotColumns ) == 1 : # ax is not 2-D
                axs = ax
            else :
                axs = ax[i]

            dataCol    = data   [ col ].to_numpy()
            gmnDataCol = gmnData[ col ].to_numpy()

            if pred_i[-1] < data.shape[0] :
                err = ComputeError( dataCol[ pred_i ], gmnDataCol )
                rho = round( err[ 'rho' ], 2 )
            else :
                rho = '' # Generated data exceeds observations

            axs[0].plot( timeData, dataCol,        label = col + ' Observed' )
            axs[1].plot( timeLib,  dataLib[ col ], label = col + ' Library'  )
            axs[1].plot( timeGMN,  gmnDataCol,
                         label = col + ' ' + parameters.mode + \
                         " [" + str( rho ) + "]" )
            axs[0].legend( loc = 'upper left', # frameon = False,
                           borderpad = 0.2, borderaxespad = 0.2 )
            axs[1].legend( loc = 'upper left', # frameon = False,
                           borderpad = 0.2, borderaxespad = 0.2 )
            i = i + 1

    #----------------------------------------------------------------------
    # Time series and state-space against first col
    #----------------------------------------------------------------------
    elif args.StatePlot or 'state' in parameters.plotType.lower() :
        col0        = plotColumns[0]
        dataCol0    = data   [ col0 ].to_numpy()
        gmnDataCol0 = gmnData[ col0 ].to_numpy()

        # plotColumns rows x 3 columns
        fig, ax = plt.subplots( len( plotColumns ), 3 )
        fig.set_size_inches( args.FigureSize )
        fig.suptitle( plotTitle + '  N=' + str( len( data ) ) + '  ' +\
                      'lib=' + str( parameters.predictionStart ) )
        fig.tight_layout( rect = ( 0, 0, 1, 0.98 ) )

        i_ax = 0
        for col in plotColumns:
            dataCol1    = data   [ col ].to_numpy()
            gmnDataCol1 = gmnData[ col ].to_numpy()

            if pred_i[-1] < data.shape[0] :
                err = ComputeError( dataCol1[pred_i], gmnDataCol1 )
                rho = round( err[ 'rho' ], 2 )
            else :
                rho = '' # Generated data exceeds observations

            if len( plotColumns ) == 1 : # ax is not 2-D
                axs = ax
            else :
                axs = ax[ i_ax ]

            # Timeseries of library + GMN prediction
            axs[0].plot( timeLib, dataLib[col], label = col + ' Lib' )
            axs[0].plot( timeGMN, gmnData[col], label = col + ' GMN' )
            axs[0].legend( loc = 'upper left',
                           borderpad = 0.2, borderaxespad = 0. )

            # Phase-space plot of observed data over the prediction indices
            axs[1].plot( dataCol1[ dataPred_i ], dataCol0[ dataPred_i ],
                         'o', markersize = 2,
                         label = "Lib: " + col0 + " ~ " + col )
            axs[1].plot( dataCol1[-1], dataCol0[-1],
                         '*', markersize = 8, color = 'red' )
            axs[1].legend( loc = 'upper center',
                           handlelength = 0, markerscale = 0, #frameon = False,
                           borderpad = 0.2, borderaxespad = 0.2 )

            # Phase-space plot of GMN data over the prediction indices
            axs[2].plot( gmnDataCol1, gmnDataCol0, 'o', markersize = 2,
                         label = "GMN: " + col0 + " ~ " + col +\
                                 " [" + str( rho ) + "]",
                         color = 'darkorange')
            axs[2].plot( gmnDataCol1[-1], gmnDataCol0[-1],
                         '*', markersize = 8, color = 'red' )
            axs[2].legend( loc = 'upper center',
                           handlelength = 0, markerscale = 0, #frameon = False,
                           borderpad = 0.2, borderaxespad = 0.2 )

            i_ax = i_ax + 1


    # Save figure file if requested
    if args.PlotFile :
        plt.savefig( args.PlotFile,
                     dpi = None, facecolor = 'w', edgecolor = 'w',
                     orientation = 'portrait', format = None,
                     transparent = False, bbox_inches = None,
                     pad_inches = 0.1, metadata = None )

    if len( parameters.plotFile ) :
        plotFile = parameters.outPath + '/' + parameters.plotFile
        plt.savefig( plotFile,
                     dpi = None, facecolor = 'w', edgecolor = 'w',
                     orientation = 'portrait', format = None,
                     transparent = False, bbox_inches = None,
                     pad_inches = 0.1, metadata = None )

    # Show pop-up if requested, user must destroy via GUI
    if args.Plot or args.StatePlot or parameters.showPlot :
        plt.show()

    if args.PlotFile or len( parameters.plotFile ) :
        plt.close( fig )
