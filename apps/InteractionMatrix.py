#! /usr/bin/env python3

# Python distribution modules
import argparse, pickle
from   datetime           import datetime
from   math               import nan
from   itertools          import combinations_with_replacement, repeat
from   concurrent.futures import ProcessPoolExecutor
from   multiprocessing    import get_context

# Community modules
import matplotlib.pyplot as plt
from   numpy  import zeros, full, corrcoef, amax, min, max, abs, maximum
from   numpy  import linspace, quantile
from   pandas import DataFrame

from   pyEDM import CCM, Simplex, ComputeError, PredictNonlinear
from   pyEDM import __version__ as pyEDM_version
from   sklearn.linear_model      import LinearRegression
from   sklearn.feature_selection import mutual_info_regression as MI
from   statsmodels.distributions.empirical_distribution import ECDF

# Local modules 
from gmn.Auxiliary import ReadDataFrame

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''Compute InteractionMatrix on all combinations of columns.

       InteractionMatrix (imatrix) rows quantify interaction for each node.
       The node in the first column of a row is the driven node, nodes in 
       other columns along the row are potential drivers. 

       Classes of Interaction Matrix
         Similitude      : rho (CC), cross map (CM), mutual information (MI)
         Shared dynamics : CCM
         Nonlinearity    : rhoDiff(?), mutual info nonlinearity (MI_NL), SMap
         Separability    : PCA, Mutual Info (distance), Clustering, 
                           multiview embedding
         Shared dynamics 
         + min redudancy : CCM -> Mutual Information (CMI)

       Avialable methods : -ccm -cmap -smap -rho -rhoDiff -mi -nl -cmi
                           [ -a = --allMethods ]
         Cross Correlation                 ( CC )    [ -rho     ]
         Simplex Cross Map                 ( CM )    [ -cmap    ]
         Convergent Cross Map              ( CCM )   [ -ccm     ]
         rho Diff = max(CM, 0) - abs(CC)             [ -rhoDiff ]
         Mutual Information                ( MI )    [ -mi      ]
         Mutual Information Non Linearity  ( MI_NL ) [ -nl      ]
         SMap nonlinearity                 ( SMap )  [ -smap    ]
         CCM -> Mutual Information         ( CMI )   [ -cmi     ]
 
       CCM:
         args.libMinFraction times N_rows is the minimum libSize
         If CCM max - min libSize > args.deltaCCM : use CCM value

       rhoDiff:
         rhoDiff = max( Simplex Cross Map, 0 ) - abs( Pearson rho )

       Mutual Information:
         Note that sklearn.feature_selection.mutual_info_regression uses a
         k-nearest neighbors MI algorithm:

         Estimating mutual information
         PHYSICAL REVIEW E 69, 066138 (2004) Kraskov, Stögbauer, Grassberger

         Mutual entropy estimators parametrized by an integer k > 1
         using kth neighbor distance statistics in the joint space. 
         Choosing small k reduces general systematic errors, while
         large k leads to smaller statistical errors. The choice of the
         particular estimator depends thus on the size of the data
         sample and on whether bias or variance is to be minimized.

       Non Linearity: 
         A mutual information approach to calculating nonlinearity.
         Stat (2015), 4: 291-303, Reginald Smith  DOI: 10.1002/sta4.96

       SMap:
          PredictNonlinear() spectrum on all columns x columns

          maxRho = max[ rho(theta) ]
          If maxRho is positive 
            deltaRho = maxRho - rho( theta = 0 )
              if   deltaRho > args.deltaSMap and maxRho > args.minRho
              then deltaRho is the nonlinearity metric

        CCM -> Mutual Information ( CMI ) 
           1) CCM : if deltaCCM > args.deltaCCM
           2) Mutual Information on positive CCM


       Efficiency is addressed by only allocating / processing
       according to values of: -ccm -cmap -smap -rho -rhoDiff -mi -nl -cmi

       StarMapFunc() is run in a pool.starmap for all methods.

       -i specifies columns from the data .csv file

       -op file output is pickled dictionary of pandas dataFrames
       -oc each matrix written to .csv 

       If EDM lib and pred not specified, use all data rows.
       Note: EDM presumes 1st data column is time.
    '''

    startTime = datetime.now()

    args = ParseCmdLine()

    if args.verbose: print( args )

    # Read data from args.dataFile
    if len( args.dataColumns ) :
        data = ReadDataFrame( args.dataFile, usecols = args.dataColumns )
    else:
        data = ReadDataFrame( args.dataFile )

    args.numCols = N = len( data.columns )
    args.numRows = data.shape[0]

    # Set lib & pred if not specified
    if len( args.lib  ) == 0 : args.lib  = [1, args.numRows]
    if len( args.pred ) == 0 : args.pred = [1, args.numRows]

    # Iterable of upper triangular of all columns x columns since CCM()
    # computes both CCM(i,j) and CCM(j,i); Start at 1 to skip first column
    crossColumns = \
        list( combinations_with_replacement( range( 1, args.numCols ), 2 ) )
    N_ = len( crossColumns )

    if args.chunksize is None :
        chunksize = maximum( 1, int( N_ / (2 * args.cores) ) )
    else :
        chunksize = args.chunksize

    mpContext = get_context( args.mpMethod )

    if args.verbose:
        print( f'{datetime.now()} ProcessPoolExecutor: {args.mpMethod} ' +\
               f'chunksize {chunksize} cores {args.cores}' )

    # ProcessPoolExecutor has no starmap(). Pass argument lists directly.
    with ProcessPoolExecutor( max_workers = args.cores,
                              mp_context = mpContext ) as exe :
        interact_ = exe.map( InteractFunc,
                             crossColumns,
                             repeat( data, N_ ),
                             repeat( args, N_ ),
                             timeout   = None,
                             chunksize = chunksize )

    # interact_ is a generator of dictionaries from InteractFunc
    interactD_ = [ _ for _ in interact_ ]

    if args.verbose:
        print( f'{datetime.now()} Finished ProcessPoolExecutor' )
        print( "Result has ", str( len( interactD_ ) ), " items." )

    # Create matrices to hold results from InteractFunc()
    CCM_mat = CM = CC = IXY = NL = rhoDiff = CMI = SMap = None
    if args.CCM      : CCM_mat = full ( ( N - 1, N - 1 ), nan )
    if args.CrossMap : CM      = full ( ( N - 1, N - 1 ), nan )
    if args.rho      : CC      = full ( ( N - 1, N - 1 ), nan )
    if args.MI       : IXY     = full ( ( N - 1, N - 1 ), nan )
    if args.MI_NL    : NL      = full ( ( N - 1, N - 1 ), nan )
    if args.CMI      : CMI     = full ( ( N - 1, N - 1 ), nan )
    if args.SMap     : SMap    = full ( ( N - 1, N - 1 ), nan )

    col1Names = [""] * ( N - 1 )
    col2Names = [""] * ( N - 1 )

    # Unpack the interactD_ list of dictionaries into matrices
    for D in interactD_ :
        if D is None :
            continue

        col1 = D['col1'] - 1  # Zero offset for time column
        col2 = D['col2'] - 1

        if args.CCM :
            CCM_mat[ col1, col2 ] = D['CCM_XY']
            CCM_mat[ col2, col1 ] = D['CCM_YX']
        if args.CrossMap :
            CM[ col1, col2 ] = D['CM_XY']
            CM[ col2, col1 ] = D['CM_YX']
        if args.rho :
            CC[ col1, col2 ] = D['CC_XY']
            CC[ col2, col1 ] = D['CC_YX']
        if args.MI :
            IXY[ col1, col2 ] = D['IXY']
            IXY[ col2, col1 ] = D['IYX']
        if args.MI_NL :
            NL [ col1, col2 ] = D['NL_XY']
            NL [ col2, col1 ] = D['NL_YX']
        if args.CMI :
            CMI[ col1, col2 ] = D['CMI_XY']
            CMI[ col2, col1 ] = D['CMI_YX']
        if args.SMap :
            SMap[ col1, col2 ] = D['SMap_XY']
            SMap[ col2, col1 ] = D['SMap_YX']

        col1Names[ col1 ] = D['column']
        col2Names[ col2 ] = D['target']

        # if "reverse" names are empty, fill in
        if not col1Names[ col2 ]: col1Names[ col2 ] = D['target']
        if not col2Names[ col1 ]: col2Names[ col1 ] = D['column']

    # Gerald defines rhoDiff = max( CM, 0 ) - abs( CC )
    if args.rhoDiff :
        CC      = abs( CC )
        CM0     = maximum( CM, zeros( N - 1 ) )
        rhoDiff = CM0 - CC

    # DataFrames of matrices
    CCM_df  = CM_df  = CC_df = IXY_df = NL_df = rhoDiff_df = None
    SMap_df = CMI_df = None

    # DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
    if args.CCM      : CCM_df     = DataFrame( CCM_mat, col2Names, col1Names )
    if args.CrossMap : CM_df      = DataFrame( CM,      col2Names, col1Names )
    if args.rho      : CC_df      = DataFrame( CC,      col2Names, col1Names )
    if args.MI       : IXY_df     = DataFrame( IXY,     col2Names, col1Names )
    if args.MI_NL    : NL_df      = DataFrame( NL,      col2Names, col1Names )
    if args.rhoDiff  : rhoDiff_df = DataFrame( rhoDiff, col2Names, col1Names )
    if args.CMI      : CMI_df     = DataFrame( CMI,     col2Names, col1Names )
    if args.SMap     : SMap_df    = DataFrame( SMap,    col2Names, col1Names )

    # Output dictionary of DataFrames
    D = { "SMap"        : SMap_df,
          "CCM"         : CCM_df,
          "CrossMap"    : CM_df,
          "MutualInfo"  : IXY_df,
          "NonLinear"   : NL_df,
          "Correlation" : CC_df,
          "rhoDiff"     : rhoDiff_df,
          "CMI"         : CMI_df }

    #-----------------------------------------
    if args.outPickleFile :
        with open( args.outPickleFile, 'wb' ) as fob:
            pickle.dump( D, fob, protocol = pickle.HIGHEST_PROTOCOL )

    #-----------------------------------------
    if args.outCSVFile :
        for key in args.methodsMap.keys() :
            value = args.methodsMap[ key ] # True or False

            if value :
                if ( key == 'SMap' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    SMap_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'CCM' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    CCM_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'CrossMap' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    CM_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'rho' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    CC_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'rhoDiff'  ):
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    rhoDiff_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'MI' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    IXY_df.round( args.precision ).to_csv( fileName )
                elif ( key == 'MI_NL' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    NL_df.round(  args.precision ).to_csv( fileName )
                elif ( key == 'CMI' ) :
                    fileName = args.outCSVFile + '_' + key + '.csv'
                    CMI_df.round( args.precision ).to_csv( fileName )

    elapsedTime = datetime.now() - startTime
    print( "Normal Exit elapsed time: ", elapsedTime )

    #-----------------------------------------
    if args.verbose :
        for key in args.methodsMap.keys() :
            value = args.methodsMap[ key ] # True or False

            if value :
                if ( key == 'SMap' ) :
                    print( "SMap:" )
                    print( SMap_df  )
                elif ( key == 'CCM' ) :
                    print( "CCM:" )
                    print( CCM_df  )
                elif ( key == 'CrossMap' ) :
                    print( "Cross Map:" )
                    print( CM_df  )
                elif ( key == 'rho' ) :
                    print( "Correlation:" )
                    print( CC_df  )
                elif ( key == 'rhoDiff'  ):
                    print( "rho Diff:" )
                    print( rhoDiff_df  )
                elif ( key == 'MI' ) :
                    print( "MI:" )
                    print( IXY_df  )
                elif ( key == 'MI_NL' ) :
                    print( "Non Linear:"  )
                    print( NL_df   )
                elif ( key == 'CMI' ) :
                    print( "CMI:" )
                    print( CMI_df  )

    #-----------------------------------------
    if args.plot :
        for key in args.methodsMap.keys() :
            value = args.methodsMap[ key ] # True or False

            if value :
                # Select matrix
                if ( key == 'SMap' ) :
                    df    = SMap_df
                    title = "S-Map Nonlinearity"
                elif ( key == 'CCM' ) :
                    df    = CCM_df
                    title = "CCM  E = " + str( args.E ) + \
                            "    row = column : col = target"
                elif ( key == 'CrossMap' ) :
                    df    = CM_df
                    title = "Simplex Cross Map  E = " + str( args.E )
                elif ( key == 'rho' ) :
                    df    = CC_df
                    title = "Cross Correlation |ρ|"
                elif ( key == 'rhoDiff' ) :
                    df    = rhoDiff_df
                    title = "CrossMap Correlation Difference (ρ-Diff)"
                elif ( key == 'MI' ) :
                    df    = IXY_df
                    title = "Mutual Info IXY"
                elif ( key == 'MI_NL' ) :
                    df    = NL_df
                    title = "MI Non Linearity  knn =" + str( args.neighbors )
                elif ( key == 'CMI' ) :
                    df    = CMI_df
                    title = "CMI  E = " + str( args.E ) + \
                            "    row = column : col = target"
                else :
                    print( "Invalid plot type. Must be one of ",
                           str( args.validPlots ) )
                    return

                plt.matshow( df )            
                plt.title( title )
                plt.xticks( range(N-1), labels = data.columns[range(1,N)],
                rotation = 'vertical' )
                plt.yticks( range(N-1), labels = data.columns[range(1,N)] )
                plt.colorbar()
                plt.show()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def InteractFunc( crossColumns, data, args ):
    '''Simplex cross map, CCM, Uncertainty coefficient, Pearsons rho,
       Mutual Information Non Linearity & Pao's rho diff on one
       pair of columns of the input data.

       Since CCM returns X:Y and Y:X, also compute pairs of all metrics
    '''

    col1 = crossColumns[ 0 ] # unpack columns from tuple
    col2 = crossColumns[ 1 ]

    if col1 == col2 :
        return  # Ignore degenerate variables

    column = data.columns[ col1 ]
    target = data.columns[ col2 ]

    x = data[ column ].values
    y = data[ target ].values

    #-------------------------------------------------------
    # EDM Cross Map via Simplex
    #-------------------------------------------------------
    CM_XY = CM_YX = None
    if args.CrossMap:
        S = Simplex( dataFrame       = data,
                     lib             = args.lib,
                     pred            = args.pred,
                     E               = args.E,
                     Tp              = args.Tp,
                     knn             = 0,
                     tau             = -1,
                     exclusionRadius = args.exclusionRadius,
                     columns         = column,
                     target          = target,
                     embedded        = False,
                     verbose         = False,
                     showPlot        = False )

        CM_XY = ComputeError( S['Observations'], S['Predictions'] )['rho']

        # Exchange columns : target for CCM_YX
        S = Simplex( dataFrame       = data,
                     lib             = args.lib,
                     pred            = args.pred,
                     E               = args.E,
                     Tp              = args.Tp,
                     knn             = 0,
                     tau             = -1,
                     exclusionRadius = args.exclusionRadius,
                     columns         = target,
                     target          = column,
                     embedded        = False,
                     verbose         = False,
                     showPlot        = False )

        CM_YX = ComputeError( S['Observations'], S['Predictions'] )['rho']

    #-------------------------------------------------------
    # EDM CCM
    #-------------------------------------------------------
    CCM_XY = CCM_YX = None
    if args.CCM:
        # Setup libSizes with two values, one small, one near N
        libMin   = max( [ 10, int( args.libMinFraction * data.shape[0] ) ] )
        libMax   = data.shape[0] - abs( args.tau ) * args.E
        libSizes = [ libMin, libMax ]

        # pyEDM 2.5.0 refactored CCM : CCM_Matrix would be better here.
        # If pyEDM 2.5.0 or greater, use legacy = True for 2.4.0 CCM
        CCM_version_ = [int(_) for _ in pyEDM_version.split('.')]
        if CCM_version_[0] != 2:
            raise RuntimeError('pyEDM version 2 required for CCM')

        if CCM_version_[1] < 5 :
            # Calling < 2.5.0 CCM
            cmap = CCM( dataFrame       = data,
                        E               = args.E,
                        Tp              = args.Tp,
                        knn             = 0,
                        tau             = -1,
                        sample          = args.sample,
                        libSizes        = libSizes,
                        exclusionRadius = args.exclusionRadius,
                        columns         = column,
                        target          = target,
                        embedded        = False,
                        verbose         = False,
                        showPlot        = False )
        else:
            # Calling >= 2.5.0 with legacy = True
            cmap = CCM( dataFrame       = data,
                        E               = args.E,
                        Tp              = args.Tp,
                        knn             = 0,
                        tau             = -1,
                        sample          = args.sample,
                        libSizes        = libSizes,
                        exclusionRadius = args.exclusionRadius,
                        columns         = column,
                        target          = target,
                        embedded        = False,
                        verbose         = False,
                        showPlot        = False,
                        legacy          = True )

        # cmap [ LibSize, column:target, target:column ] x 2 rows
        # delta is CCM at large libSize - CCM at small libSize
        CCM_XY = CCM_YX = 0 # no convergence
        deltaCCM_XY = max( cmap.iloc[1,1] - cmap.iloc[0,1], 0 )
        deltaCCM_YX = max( cmap.iloc[1,2] - cmap.iloc[0,2], 0 )
        if deltaCCM_XY > args.deltaCCM :
            CCM_XY = cmap.iloc[1,1] # Large library value
        if deltaCCM_YX > args.deltaCCM :
            CCM_YX = cmap.iloc[1,2] # Large library value

    #-------------------------------------------------------
    # Linear correlation 
    #-------------------------------------------------------
    CC_XY = CC_YX = None
    if args.rho:
        CC_XY = corrcoef( x, y )[0,1]
        CC_YX = CC_XY

    #-------------------------------------------------------
    # Non Linearity based on Mutual Information
    #-------------------------------------------------------
    IXY = IYX = NonLinear_XY = NonLinear_YX = None
    if args.MI :
        # Step 1: Mutual information of original variables.
        IXY = MI( x.reshape(-1,1), y,
                  discrete_features = False, n_neighbors = args.neighbors,
                  copy = True, random_state = None )[0]

        IYX = MI( y.reshape(-1,1), x,
                  discrete_features = False, n_neighbors = args.neighbors,
                  copy = True, random_state = None )[0]

    if args.MI_NL :
        # Step 2: Least-squares regression predicting Y given X.
        #         Y_ = predictions  residuals z = Y - Y_
        LM_XY = LinearRegression( copy_X = True, n_jobs = None )
        LM_XY.fit( x.reshape(-1,1), y )
        z_XY = y - LM_XY.predict( x.reshape(-1,1) )

        LM_YX = LinearRegression( copy_X = True, n_jobs = None )
        LM_YX.fit( y.reshape(-1,1), x )
        z_YX = x - LM_YX.predict( y.reshape(-1,1) )

        # Step 3: Analysis of residuals.
        # van der Waerden normal quantile transform
        # Map CDF of the linear regression residuals onto the CDF of
        # the dependent variable to ensure the mapped residual CDF and PDF
        # match those of the dependent variable and have the same entropy.
        Gz_XY     = ECDF( z_XY )
        zVals_XY  = linspace( min( y ), max( y ), len( y ) )
        GzProb_XY = Gz_XY( zVals_XY )
        yPrime_XY = quantile( y, GzProb_XY )

        Gz_YX     = ECDF( z_YX )
        zVals_YX  = linspace( min( x ), max( x ), len( x ) )
        GzProb_YX = Gz_YX( zVals_YX )
        yPrime_YX = quantile( x, GzProb_YX )

        # Step 4: Calculate mutual information of X and Y'
        #         Ixy' represents the mutual information of X and Y minus
        #         their linear dependence.   Note: Ixy >= Ixy'.
        IXYp = MI( x.reshape(-1,1), yPrime_XY,
                   discrete_features = False, n_neighbors = args.neighbors,
                   copy = True, random_state = None )[0]

        IYXp = MI( y.reshape(-1,1), yPrime_YX,
                   discrete_features = False, n_neighbors = args.neighbors,
                   copy = True, random_state = None )[0]

        # The overall proportion of linear dependence to all dependence
        # between X and Y is Linear = 1 - Ixy' / Ixy
        #   Linear = 0:  No linear dependence between variables,
        #                but does not mean MI = 0. If MI > 0, it indicates
        #                all dependence is completely nonlinear.
        #   Linear = 1 : The relationship is entirely linear.
        if ( IXYp > IXY ) :
            IXYp = 0  # Invalid model... set for 0 NonLinear

        if ( IXY == 0 ) :  # X & Y are independent
            NonLinear_XY = nan
            Linear_XY    = nan
        else :
            NonLinear_XY = IXYp / IXY 
            Linear_XY    = 1 - NonLinear_XY

        if ( IYXp > IYX ) :
            IYXp = 0  # Invalid model... set for 0 NonLinear

        if ( IYX == 0 ) :  # X & Y are independent
            NonLinear_YX = nan
            Linear_YX    = nan
        else :
            NonLinear_YX = IYXp / IYX
            Linear_YX    = 1 - NonLinear_YX

    #-------------------------------------------------------
    # CCM -> MI
    #-------------------------------------------------------
    CMI_XY = CMI_YX = None
    if args.CMI:
        if CCM_XY > 0 :
            CMI_XY = IXY
        if CCM_YX > 0 :
            CMI_YX = IYX

    #-------------------------------------------------------
    # SMap NL
    #-------------------------------------------------------
    SMap_XY = SMap_YX = None
    if args.SMap :
        SMap_XY = PredictNL( column, target, args, data )
        SMap_YX = PredictNL( target, column, args, data )

    #-------------------------------------------------------
    result = { 'col1'    : col1,
               'col2'    : col2,
               'column'  : column,
               'target'  : target,
               'CCM_XY'  : CCM_XY,
               'CCM_YX'  : CCM_YX,
               'CM_XY'   : CM_XY,
               'CM_YX'   : CM_YX,
               'CC_XY'   : CC_XY,
               'CC_YX'   : CC_YX,
               'IXY'     : IXY,
               'IYX'     : IYX,
               'NL_XY'   : NonLinear_XY,
               'NL_YX'   : NonLinear_YX,
               'CMI_XY'  : CMI_XY,
               'CMI_YX'  : CMI_YX,
               'SMap_XY' : SMap_XY,
               'SMap_YX' : SMap_YX }

    return result

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def PredictNL( column, target, args, data ):
    '''SMap PredictNonlinear'''

    if len( args.theta ) :
        # theta specified
        numProcess = min( [ len( args.theta ), args.cores ] ) # numpy min
    else :
        # theta not specified, pyEDM default has 15 values
        numProcess = min( [ 15, args.cores ] )

    #-------------------------------------------------------
    #    DataFrame with columns theta and rho. 
    #-------------------------------------------------------
    DF = PredictNonlinear( dataFrame       = data,
                           lib             = args.lib,
                           pred            = args.pred,
                           theta           = args.theta,
                           E               = args.E,
                           tau             = args.tau,
                           exclusionRadius = args.exclusionRadius,
                           Tp              = args.Tp,
                           columns         = column,
                           target          = target,
                           embedded        = False,
                           verbose         = False,
                           numProcess      = numProcess,
                           showPlot        = False )

    rho = DF[ 'rho' ]

    # Does theta : rho exhibit nonlinearity?
    #   maxRho - thetaRho (theta = 0?) greater than args.delta ?
    maxRho = amax( rho )
    
    # First, ensure minimum rho
    if maxRho > args.minRho :
        deltaRho = maxRho - rho[0]
    else :
        deltaRho = 0

    if deltaRho > args.deltaSMap :
        nonLinear = deltaRho
    else :
        nonLinear = 0

    if args.verbose :
        print( '(', column, ':', target, ')',
               ' deltaRho: ',  round( deltaRho, 4 ),
               ' nonLinear: ', round( nonLinear, 4 ), flush = True )
        print( rho, flush = True )
        print( flush = True )
        print()

    return nonLinear

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = argparse.ArgumentParser( description = 'Mutual Information' )

    parser.add_argument('-cr', '--cores',
                        dest   = 'cores', type = int, 
                        action = 'store', default = 8,
                        help = 'Multiprocessing cores.')

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = None,
                        help = 'ProcessPool chunksize')

    parser.add_argument('-oc', '--outCSVFile',
                        dest   = 'outCSVFile', type = str,
                        action = 'store',   default = None,
                        help = 'Output file (.csv).')

    parser.add_argument('-op', '--outPickleFile',
                        dest   = 'outPickleFile', type = str,
                        action = 'store',   default = None,
                        help = 'Binary output file (.pkl).')

    parser.add_argument('-d', '--dataFile',
                        dest   = 'dataFile', type = str,
                        action = 'store', default = "../data/TestData_ABCD.csv",
                        help = 'Data file (csv).')

    parser.add_argument('-i', '--i_columns',
                        dest   = 'dataColumns', nargs = '+', type = int, 
                        action = 'store', default = [],
                        help = 'Data column indices.')

    parser.add_argument('-n', '--neighbors',
                        dest   = 'neighbors', type = int, 
                        action = 'store', default = 3,
                        help = 'MI neighbors.')

    parser.add_argument('-E', '--E',
                        dest   = 'E', type = int, 
                        action = 'store', default = 2,
                        help = 'Simplex & CCM E.')

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int, 
                        action = 'store', default = 0,
                        help = 'Simplex Tp.')

    parser.add_argument('-t', '--tau',
                        dest   = 'tau', type = int, 
                        action = 'store', default = -1,
                        help = 'Simplex tau.')

    parser.add_argument('-x', '--exclusionRadius',
                        dest   = 'exclusionRadius', type = int, 
                        action = 'store', default = 0,
                        help = 'Simplex exclusionRadius.')

    parser.add_argument('-th', '--theta',
                        nargs  = '+',
                        dest   = 'theta', type = float,
                        action = 'store', default = [],
                        help = 'SMap theta list.')

    parser.add_argument('-D', '--deltaSMap',
                        dest   = 'deltaSMap', type = float, 
                        action = 'store', default = 0.05,
                        help = 'delta SMap rho for non linear.')

    parser.add_argument('-m', '--minRho',
                        dest   = 'minRho', type = float, 
                        action = 'store', default = 0.2,
                        help = 'minimum rho for valid prediction.')

    parser.add_argument('-l', '--library',
                        nargs  = '+',
                        dest   = 'lib', type = int, 
                        action = 'store', default = [],
                        help = 'Simplex lib.')

    parser.add_argument('-p', '--prediction',
                        nargs  = 2,
                        dest   = 'pred', type = int, 
                        action = 'store', default = [],
                        help = 'Simplex pred.')

    parser.add_argument('-s', '--sample',
                        dest   = 'sample', type = int, 
                        action = 'store', default = 20,
                        help = 'CCM samples.')

    parser.add_argument('-C', '--deltaCCM',
                        dest   = 'deltaCCM', type = float, 
                        action = 'store', default = 0.05,
                        help = 'delta CCM for convergence.')

    parser.add_argument('-lm', '--libMinFraction',
                        dest   = 'libMinFraction', type = float, 
                        action = 'store', default = 0.05,
                        help = 'Fraction of data length for min CCM libSize.')

    parser.add_argument('-pr', '--precision',
                        dest   = 'precision', type = int, 
                        action = 'store', default = 6,
                        help = 'CSV file numerical precision.')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true',  default = False,
                        help = 'Plot matrix of specified methods')

    parser.add_argument('-a', '--allMethods',
                        dest   = 'allMethods',
                        action = 'store_true', default = False,
                        help = 'Compute all interaction types.')

    parser.add_argument('-smap', '--smap',
                        dest   = 'SMap',
                        action = 'store_true', default = False,
                        help = 'SMap.')

    parser.add_argument('-ccm', '--ccm',
                        dest   = 'CCM',
                        action = 'store_true', default = False,
                        help = 'CCM.')

    parser.add_argument('-cmap', '--CrossMap',
                        dest   = 'CrossMap',
                        action = 'store_true', default = False,
                        help = 'CrossMap.')

    parser.add_argument('-rho', '--rho',
                        dest   = 'rho',
                        action = 'store_true', default = False,
                        help = 'rho.')

    parser.add_argument('-rhoDiff', '--rhoDiff',
                        dest   = 'rhoDiff',
                        action = 'store_true', default = False,
                        help = 'rhoDiff.')

    parser.add_argument('-cmi', '--cmi',
                        dest   = 'CMI',
                        action = 'store_true', default = False,
                        help = 'CCM -> MI.')

    parser.add_argument('-mi', '--MI',
                        dest   = 'MI',
                        action = 'store_true', default = False,
                        help = 'MI.')

    parser.add_argument('-nl', '--MI_NL',
                        dest   = 'MI_NL',
                        action = 'store_true', default = False,
                        help = 'MI_NL.')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'verbose.')

    args = parser.parse_args()

    if args.allMethods :
        args.SMap = args.CCM = args.CrossMap = args.rho =\
        args.rhoDiff = args.MI = args.MI_NL = args.CMI = True

    else :
        # Validity checks
        # rhoDiff requires cmap and rho
        if args.rhoDiff :
            args.CrossMap = True
            args.rho      = True

        # MI_NL requires MI
        if args.MI_NL :
            args.MI = True

        # CMI requires CCM and MI
        if args.CMI :
            args.CCM = args.MI = True

    # Collect method switches into a dict
    args.methodsMap = { 'SMap'    : args.SMap,
                        'CCM'     : args.CCM,
                        'CrossMap': args.CrossMap,
                        'rho'     : args.rho,
                        'rhoDiff' : args.rhoDiff,
                        'MI'      : args.MI,
                        'MI_NL'   : args.MI_NL,
                        'CMI'     : args.CMI }

    if not any( args.methodsMap.values() ) :
        raise RuntimeError( "No method specified. Options: " +\
                            "-a -smap -ccm -cmap -rho -rhoDiff -mi -nl -cmi" )

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
