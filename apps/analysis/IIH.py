#! /usr/bin/env python3

# Python distribution modules
from argparse import ArgumentParser

# Community modules
from pandas   import read_csv, read_feather, DataFrame
from numpy    import diff, histogram, insert, zeros, nonzero

import matplotlib.pyplot as plt

# Local modules

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def IIH( var, varName = '', upper = 0.1, bins = 15, binRange = None,
         density = True, plot = False, verbose = False ):
    '''Interevent Interval Histogram (IIH)

    Estimate IIH on var : numpy array
    var_array holds a binary representation of event start times
    Variable values >= upper threshold are set to 1 in var_array
    First differences of var_array equal 0-1 = -1 at event start
    Indices of event starts are differenced to compute interevent intervals
    histogram is computed on interevent intervals

    Return dict( 'IIH'       : iih counts|density,
                 'bins'      : bins,
                 'intervals' : ii )
    '''

    #var_lower = var < lower # boolean mask
    var_array = zeros( len( var ), dtype = int )
    var_upper = var >= upper      # boolean mask
    var_array[var_upper] = 1      # set var >= upper indices = 1
    var_diff  = diff( var_array ) # var_diff values of -1 are event start
    var_diff  = insert( var_diff, 0, 0 ) # insert 0 first diff to match var len
    
    # indices of var_diff == -1 mark event starts
    i_ = nonzero( var_diff == -1 )[0] # nonzero() returns tuple...
    ii = diff( i_ ) # interevent intervals

    if len( ii ) < 5 or \
       (not binRange is None and ii.min() > binRange[1]):
        # Not enough interevent intervals or beyond binRange
        iih, bins = None, None
    else :
        iih, bins = histogram( ii, bins=bins, range=binRange, density=density )

    if verbose :
        print( f'{varName} ----------------------' )
        print( f'ii {ii}' )
        print( f'iih {iih.round(3)}' )
        print( f'bins {bins.round(1)}' )

    if plot :
        # Use matplotlib hist()
        fig, axs = plt.subplots(1, 1, tight_layout=True)
        axs.set_title ( f'IIH {varName}' )
        axs.set_xlabel( 'Interevent interval' )
        axs.set_ylabel( '' )
        
        iih_, bins_, patches = axs.hist( ii, bins = bins,
                                         range = binRange, density = density )
        plt.show()

    return { 'IIH':iih, 'bins':bins, 'intervals':ii }

#--------------------------------------------------------------
#--------------------------------------------------------------
def IIH_CmdLine():
    '''Command line wrapper for IIH()'''
    args = ParseCmdLine()

    # Read data with node columns
    if '.csv' in args.dataFile[-4:] :
        df = read_csv( args.dataFile )
    elif '.feather' in args.dataFile[-8:] :
        df = read_feather( args.dataFile )

    var = df[ args.variable ]

    D = IIH( var, varName = args.variable, upper = args.upper,
             bins = args.bins, binRange = args.binRange,
             density = args.density, plot = args.plot,
             verbose = args.verbose )

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseCmdLine( argv = None ):
    '''Pass argv = [] to return default args
       Pass argv = sys.argv[1:] to get command line args via API'''

    parser = ArgumentParser( description = 'IIH' )

    parser.add_argument('-d', '--dataFile',
                        dest    = 'dataFile', type = str, 
                        action  = 'store',
    default = '~/Research/Data/LorenFrankLab/J16_2021-06-05_epoch_8_1_Hz.csv',
                        help    = 'Input data file.')

    parser.add_argument('-var', '--variable',
                        dest    = 'variable', type = str, 
                        action  = 'store',
                        default = 'CA1_51',
                        help    = 'variable.')

    parser.add_argument('-u', '--upper',
                        dest    = 'upper', type = float, 
                        action  = 'store',
                        default = 0.1,
                        help    = 'Amplitude upper threshold')

    #parser.add_argument('-l', '--lower',
    #                    dest    = 'lower', type = float, 
    #                    action  = 'store',
    #                    default = 0.,
    #                    help    = 'Amplitude lower threshold')

    #parser.add_argument('-w', '--width',
    #                    dest    = 'width', type = int, 
    #                    action  = 'store',
    #                    default = 1,
    #                    help    = 'Subsample data width (points)')

    parser.add_argument('-b', '--bins',
                        dest    = 'bins', type = int, 
                        action  = 'store',
                        default = 15,
                        help    = 'histogram number of bins')

    parser.add_argument('-r', '--binRange', nargs = 2,
                        dest    = 'binRange', type = int, 
                        action  = 'store',
                        default = None,
                        help    = 'histogram value range')

    parser.add_argument('-D', '--density',
                        dest    = 'density',
                        action  = 'store_true',
                        default = False,
                        help    = 'histogram density instead of counts.')

    parser.add_argument('-p', '--plot',
                        dest    = 'plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot.')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true',
                        default = False )

    args = parser.parse_args( argv ) # if argv is None : default = sys.argv[1:]

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
#----------------------------------------------------------------------------
if __name__ == "__main__":
    IIH_CmdLine()
