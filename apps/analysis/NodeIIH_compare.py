#! /usr/bin/env python3

# Python distribution modules
from argparse import ArgumentParser
from pickle   import load as pickle_load

# Community modules
from pandas import read_csv, read_feather, DataFrame
from numpy  import zeros

import matplotlib.pyplot as plt

# Local modules
from IIH       import IIH
from KStest    import KStest
from KLdiverge import KLdiverge

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def NodeIIH( gmnNetwork = None, # dict() from CreateNetwork gmn['Map']
             generated  = None, # DataFrame
             observed   = None, # DataFrame
             genUpper   = 0.1,  # IIH event threshold
             obsUpper   = 0.1,  # IIH event threshold
             bins       = 15,   # IIH histogram 
             binRange   = None, # None : IIH histogram will use a.min - a.max
             pvalueThreshold = 0.01,
             returnPercent   = False, # return percent nodes KS > pvalue
             computeKL       = False, # compute Kullback-Leibler distance
             plot = False, verbose = False ):
    '''Compare Interevent Intervals of generated and observed data at GMN nodes

    Two comparison metrics:
       1) Kolmorgorov-Smirnov test  <- interevent intervals
       2) Kullback-Leibler distance <- IIH density

    generated & observed DataFrames are subset to equal number of rows
    generated & observed are normalized to [0,1]
    
    Interevent intervals and IIH are computed in IIH() stored in dicts
    D_gen_IIH & D_obs_IIH. genUpper, obsUpper are amplitude thresholds
    to identify event starts. 

    if returnPercent True : return percentNode with pvalue > pvalueThreshold
    else : return DataFrame( KS_p = KS_pvalue, KS = KS_values, [KL = KL_values] )
    '''
    # Validation ---------------------------------------------------------
    if not isinstance( gmnNetwork, dict ) :
        raise RuntimeError( 'NodeIIH(): gmnNetwork is not dict' )
    if not isinstance( generated, DataFrame ) :
        raise RuntimeError( 'NodeIIH(): generated is not DataFrame' )
    if not isinstance( observed, DataFrame ) :
        raise RuntimeError( 'NodeIIH(): observed is not DataFrame' )
    if not all( [n in generated.columns for n in gmnNetwork.keys()] ) :
        raise RuntimeError( 'NodeIIH(): GMN nodes missing from generated' )
    if not all( [n in observed.columns for n in gmnNetwork.keys()] ) :
        raise RuntimeError( 'NodeIIH(): GMN nodes missing from observed' )
    if verbose :
        print( f'    gmnNetwork {len(gmnNetwork)} nodes' )
        print( f'    observed {observed.shape}   generated {generated.shape}' )

    class emptyKS:
        def __init__( self ) :
            self.statistic          = None
            self.pvalue             = None
            self.statistic_location = None
            self.statistic_sign     = None

    # Subset to match generated & observed number of observations -----------
    N_gen = generated.shape[0]
    N_obs = observed.shape[0]
    if N_gen > N_obs :
        generated = generated.iloc[:N_obs,:]
    elif N_obs > N_gen :
        observed = observed.iloc[:N_gen,:]
    if verbose :
        print( f'    observed {observed.shape}   generated {generated.shape}' )

    # Normalize [0,1] --------------------------------
    generated = ( generated - generated.min() ) / \
                ( generated.max() - generated.min() )
    observed =  ( observed - observed.min() ) / \
                ( observed.max() - observed.min() )

    # for each node compute {IIH, bins, intervals} of generated and observed
    # If no ii found {IIH=None, bins=None, intervals=[]}
    D_gen_IIH = dict()
    D_obs_IIH = dict()
    D_IIH_KS  = dict()
    D_data_KS = dict()
    D_IIH_KL  = dict()

    for node in gmnNetwork.keys() :
        D_gen_IIH[ node ] = IIH( generated[node], varName = node,
                                 upper = genUpper, bins = bins,
                                 binRange = binRange )

        D_obs_IIH[ node ] = IIH( observed[node], varName = node,
                                 upper = obsUpper, bins = bins,
                                 binRange = binRange )

        if not D_gen_IIH[ node ]['IIH'] is None and \
           not D_obs_IIH[ node ]['IIH'] is None :
            # Kolmorgorov - Smirnov statistic on generated : observed intervals
            D_IIH_KS[ node ] = KStest( D_gen_IIH[ node ]['intervals'],
                                       D_obs_IIH[ node ]['intervals'],
                                       f'gen_{node}', f'obs_{node}' )
            if computeKL :
                # Kullback-Leibler distance : density IIH
                D_IIH_KL[ node ] = KLdiverge( D_gen_IIH[ node ]['IIH'],
                                              D_obs_IIH[ node ]['IIH'],
                                              f'gen_{node}', f'obs_{node}' )
        else :
            D_IIH_KS[ node ] = emptyKS()

        D_data_KS[ node ] = KStest( observed[node], generated[node],
                                    'obs_'+node, 'gen_'+node )

    # Output DataFrame
    IIH_KS_values  = [ ks.statistic for ks in D_IIH_KS.values() ]
    IIH_KS_pvalue  = [ ks.pvalue    for ks in D_IIH_KS.values() ]
    data_KS_value  = [ ks.statistic for ks in D_data_KS.values() ]
    data_KS_pvalue = [ ks.pvalue    for ks in D_data_KS.values() ]
    if computeKL :
        KL_values = D_IIH_KL.values()

        df = DataFrame( dict( IIH_KS_pval  = IIH_KS_pvalue,
                              IIH_KS       = IIH_KS_values, 
                              data_KS_pval = data_KS_pvalue,
                              data_KS      = data_KS_value,
                              KL           = KL_values ),
                        index = D_IIH_KL.keys() )
    else :
        df = DataFrame( dict( IIH_KS_pval  = IIH_KS_pvalue,
                              IIH_KS       = IIH_KS_values, 
                              data_KS_pval = data_KS_pvalue,
                              data_KS      = data_KS_value ),
                        index = D_IIH_KS.keys() )

    df_ = df.loc[ df['IIH_KS_pval'] > pvalueThreshold, : ]
    percentNode = 100 * df_.shape[0] / df.shape[0]
    
    if verbose :
        print( f'{df_.shape[0]} of {df.shape[0]} nodes {percentNode:.1f}% ' +\
               f'with IIH K-S p-value > {pvalueThreshold}' )
        print( df_ )

    if plot :
        #df.plot( y = df.columns, subplots = True, lw = 3, figsize = (12,8) )
        axes = df.plot( y = ['IIH_KS_pval','IIH_KS'], style = '.-',
                        subplots = True,
                        markersize = 18, linewidth = 3, figsize = (12,6) )
        axes[0].axhline( y = pvalueThreshold, color = 'black', linestyle = '-',
                         linewidth = 1, label = str(pvalueThreshold) )

        for ax in axes :
            ax.tick_params(axis='both', labelsize=14)
            ax.legend( fontsize = 14 )
        plt.subplots_adjust( hspace = 0 )
        plt.tight_layout()
        plt.show()

    if returnPercent :
        return percentNode
    else :
        return df

#--------------------------------------------------------------
#--------------------------------------------------------------
def NodeIIH_CmdLine():
    '''Command line wrapper for NodeIIH()'''
    args = ParseCmdLine()

    # Read GMN network
    with open( args.gmnFile, 'rb' ) as f:
        gmnNetwork = pickle_load( f )

    if not isinstance( gmnNetwork, dict ) :
        raise RuntimeError( 'NodeIIH_CmdLine(): gmnNetwork is not dict' )

    # Read generated data
    if '.csv' in args.generatedFile[-4:] :
       generated = read_csv( args.generatedFile )
    elif '.feather' in args.generatedFile[-8:] :
       generated = read_feather( args.generatedFile )

    # Read observed data
    if '.csv' in args.observedFile[-4:] :
       observed = read_csv( args.observedFile )
    elif '.feather' in args.observedFile[-8:] :
       observed = read_feather( args.observedFile )

    D = NodeIIH( gmnNetwork      = gmnNetwork['Map'],
                 generated       = generated,
                 observed        = observed,
                 genUpper        = args.genUpper,
                 obsUpper        = args.obsUpper,
                 bins            = args.bins,
                 binRange        = args.binRange,
                 pvalueThreshold = args.pvalueThreshold,
                 returnPercent   = args.returnPercent,
                 computeKL       = args.computeKL,
                 plot            = args.plot,
                 verbose         = args.verbose )

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseCmdLine( argv = None ):
    '''Pass argv = [] to return default args
       Pass argv = sys.argv[1:] to get command line args via API'''

    parser = ArgumentParser( description = 'IIHNode_compare' )

    parser.add_argument('-nf', '--gmnFile',
                        dest    = 'gmnFile', type = str, 
                        action  = 'store',
    default = '../../../out/Rat_J16/CCM_iMatrix/GMNet_PosX_EDim_T0.6.pkl',
                        help    = 'GMN network file.')

    parser.add_argument('-gf', '--generatedFile',
                        dest    = 'generatedFile', type = str, 
                        action  = 'store',
    default='../../../out/Rat_J16/DataOut/RatJ16_PosX_EDim_T0.6_DataOut.feather',
                        help    = 'GMN generated data file.')

    parser.add_argument('-of', '--observedFile',
                        dest    = 'observedFile', type = str, 
                        action  = 'store',
    default = '~/Research/Data/LorenFrankLab/J16_2021-06-05_epoch_8_1_Hz.csv',
                        help    = 'Input observed data file.')

    parser.add_argument('-gu', '--genUpper',
                        dest    = 'genUpper', type = float, 
                        action  = 'store',
                        default = 0.1,
                        help    = 'Amplitude upper threshold')

    parser.add_argument('-ou', '--obsUpper',
                        dest    = 'obsUpper', type = float, 
                        action  = 'store',
                        default = 0.1,
                        help    = 'Amplitude upper threshold')

    parser.add_argument('-t', '--pvalueThreshold',
                        dest    = 'pvalueThreshold', type = float, 
                        action  = 'store',
                        default = 0.01,
                        help    = 'p-value threshold')

    parser.add_argument('-b', '--bins',
                        dest    = 'bins', type = int, 
                        action  = 'store',
                        default = 15,
                        help    = 'histogram number of bins')

    parser.add_argument('-br', '--binRange', nargs = 2,
                        dest    = 'binRange', type = int, 
                        action  = 'store',
                        default = None,
                        help    = 'histogram value range')

    parser.add_argument('-r', '--returnPercent',
                        dest    = 'returnPercent',
                        action  = 'store_true',
                        default = False,
                        help    = 'returnPercent nodes w/ pvalue > threshold.')

    parser.add_argument('-KL', '--computeKL',
                        dest    = 'computeKL',
                        action  = 'store_true',
                        default = False,
                        help    = 'compute Kullback-Leibler distance.')

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
    NodeIIH_CmdLine()
