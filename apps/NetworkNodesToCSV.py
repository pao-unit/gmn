#! /usr/bin/env python3

# Python distribution modules
from   argparse import ArgumentParser
from   glob     import glob
import pickle

# Community modules
from pandas import DataFrame

# Local modules 

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def main():
    '''Given a directory with CreateNetwork.py .pkl files 
       For each network.pkl read the node names
       Write a .csv with each column a list of node names for a network
       The network file name is the column label.
    '''

    args = ParseCmdLine()

    pklFiles = glob( args.directory + '*.pkl' )

    networkDict = {}

    # for files ending in .pkl, load the network object
    for fileName in pklFiles :

        with open( fileName, 'rb' ) as f :
            NetworkGraphDict = pickle.load( f )
            Graph      = NetworkGraphDict[ 'Graph' ]
            NetworkMap = NetworkGraphDict[ 'Map'   ] # Not used

            networkDict[ fileName ] = [ n for n in Graph.nodes ]

    # DataFrame() must have equal number of rows for each column
    N = max( [len(nodes) for nodes in networkDict.values() ] )

    print( f'Max number of nodes is {N}' )
    
    # Create DataFrame
    df = DataFrame( columns = networkDict.keys(),
                    index = [n for n in range( N )] )

    for name, nodes in networkDict.items() :
        for i,node in zip( range(len(nodes)), nodes ) :
            df.loc[ i, name ] = node

    if args.DEBUG :
        print( df )
    
    df.to_csv( args.outputFile, index = False )

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseCmdLine():

    parser = ArgumentParser( description = 'Network Nodes to CSV' )

    parser.add_argument('-d', '--directory',
                        dest    = 'directory', type = str, 
                        action  = 'store',
                        default = './',
                        help    = 'Directory of network (.pkl) files.')

    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = 'nodes.csv',
                        help    = '.csv output file')

    parser.add_argument('-D', '--DEBUG',
                        dest   = 'DEBUG', # type = bool, 
                        action = 'store_true', default = False )

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
