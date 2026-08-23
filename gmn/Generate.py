
# Community modules
from pandas import concat
from numpy  import arange
from pyEDM  import Embed

# Local modules
from .Common import *

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def Generate( self, lastDataOut ):
    '''Node method
       Called from GMN Generate() in the Network Node loop'''

    if self.args.DEBUG :
        print( '-----> Node:Generate() : ', self.FunctionType.name, self.name )
        print( self.data.tail( 2 ) );
        print( 'lastDataOut:' ); print( lastDataOut, flush = True )

    # Kernel fast path : taken BEFORE the growing-DataFrame append, since
    # eliminating that per-step concat is a main kernel win. Eligible
    # default-path Simplex nodes advance only their rolling buffer and
    # project one step ; no pyEDM / pandas / kd-tree per step.
    if self.KernelLib is not None :
        from .SimplexKernel import Generate as KernelGenerate

        # Advance the rolling buffer with this node's input columns from
        # the appended t-1 row ( except on time step 0 ). Index the row
        # by CACHED integer positions into a plain numpy array : no
        # pandas / Arrow column selection, which otherwise dominates the
        # per-step cost at scale ( pyarrow.take, string-index resolve ).
        if lastDataOut is not None :
            rowVals = lastDataOut.to_numpy()[ 0 ]
            newRaw  = rowVals[ self.KernelColPos ]
            self.KernelLib.Append( newRaw )

        # One kernel step : pred vector -> knn -> projection scalar.
        return KernelGenerate( self.KernelLib )

    # Append new data to end of node data : except on time step 0
    # Do not insert via .loc[] : stackoverflow.com/questions/57000903/
    if not ( lastDataOut is None ):
        self.data = concat( [ self.data, lastDataOut ] )

    # Local References for convenience and readability
    Parameters = self.Parameters
    data       = self.data

    if self.args.DEBUG :
        print( '  Appended data : shape: ', data.shape );
        print( data.tail( 3 ) ); print('', flush = True)

    #--------------------------------------------------------------------
    if self.FunctionType.value == FunctionType.Simplex.value :

        # If lib or pred are set in Parameters, use them
        # otherwise lib = [1, libEnd_i], pred = [N-1, N] N = data.shape[0]
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else:
            lib = "1 %d" % self.libEnd_i # Constant... for now

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )

        # kdWorkers controls scipy KDTree.query threading inside Simplex :
        # set from backend so outer (node) and inner (query) parallelism do
        # not oversubscribe. -1 = all cores (serial node loop); 1 = single
        # (parallel node loop supplies concurrency). See GMN.Generate().
        S = self.Function( dataFrame       = data,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = Parameters.columns,
                           target          = Parameters.target,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps,
                           kdWorkers       = self.kdWorkers )

        val = S['Predictions'].iloc[-1]

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.SMap.value :

        # JP TODO: Add sklearn solver import/functions...
        # Convert Parameters.solver to None if empty
        # if len( Parameters.solver ) == 0 or Parameters.solver.isspace():
        #     self.Parameters.solver = None

        # SMap multivariate requires embedded = True
        # Embed data to E, tau
        df = Embed( dataFrame = data, E = Parameters.E,
                    tau = Parameters.tau, columns = Parameters.columns )

        # Remove leading NaN from the time shift
        offset = ( Parameters.E - 1 ) * abs( Parameters.tau )
        df = df.iloc[ offset : ]

        # Create & insert bogus time column for SMap dataFrame
        timeColumn = arange( df.shape[0] )
        df.insert( 0, 'time', timeColumn, allow_duplicates = True )

        # If lib or pred are set in Parameters, use them
        # otherwise lib = [1, libEnd_i], pred = [N-1, N] N = df.shape[0]
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else :
            lib  = "1 %d" % self.libEnd_i # Constant... for now

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )

        # Note: dataFrame = df, columns, target from embedding
        S = self.Function( dataFrame       = df,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           theta           = Parameters.theta,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = df.columns[1:-1],
                           target          = Parameters.target + '(t-0)',
                           solver          = None, # Parameters.solver,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = S['predictions']['Predictions'].iloc[-1]

        # SMap can diverge if E, tau are not appropriate
        if abs( val ) > 1E15 :
            raise RuntimeError( "Generate:SMap() Node: " + self.name +\
                                "   Divergence detected (>1E15)" )

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.kedmSimplex.value :

        data_ = data[ Parameters.columns ] # DataFrame; includes target

        # Get lib, pred and target for kedm simplex
        # JP kedm seems to core if not enough pred rows
        targetLen = Parameters.E * abs( Parameters.tau )
        pred    = data_.iloc[ -targetLen : ]    # targetLen rows before end
        library = data_.iloc[ : self.libEnd_i ] # first to libEnd_i rows
        target  = library[ Parameters.target ]  # same rows as library

        S = self.Function( lib    = library.to_numpy(), # ndarray for kedm
                           pred   = pred.to_numpy(),    # ndarray for kedm
                           target = target.to_numpy(),  # ndarray for kedm
                           E      = Parameters.E,
                           tau    = abs( Parameters.tau ),
                           Tp     = Parameters.Tp )

        val = S[-1] # numpy.ndarray

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.kedmSMap.value :

        # JP: kedm smap is univariate, force to self.name node I/O
        data_ = data[ self.name ] # Series

        # Get lib, pred and target for kedm smap
        # JP kedm seems to core if not enough pred data
        targetLen = Parameters.E * abs( Parameters.tau )
        pred    = data_.iloc[ -targetLen : ]    # targetLen rows before end
        library = data_.iloc[ : self.libEnd_i ] # first to libEnd_i rows
        target  = library                       # same rows as library

        S = self.Function( lib    = library.to_numpy(), # ndarray for kedm
                           pred   = pred.to_numpy(),    # ndarray for kedm
                           target = target.to_numpy(),  # ndarray for kedm
                           E      = Parameters.E,
                           tau    = abs( Parameters.tau ),
                           Tp     = Parameters.Tp,
                           theta  = Parameters.theta )

        val = S[-1] # numpy.ndarray

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.Linear.value :

        X = data[ Parameters.columns ]
        #if len( Parameters.columns ) > 1 :
        #    X  = X.drop( self.name, axis = 'columns' )
        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        lr = self.Function()
        lr.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = lr.predict( nextX )
        val   = pred[ 0 ]

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.SVR.value :

        X = data[ Parameters.columns ]
        #if len( Parameters.columns ) > 1 :
        #    X  = X.drop( self.name, axis = 'columns' )
        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        svr = self.Function( kernel = 'rbf', gamma = 'scale', tol = 0.001,
                             epsilon = 0.1, shrinking = True, cache_size = 200,
                             verbose = False, max_iter = -1 )
        svr.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = svr.predict( nextX )
        val   = pred[ 0 ]

    #--------------------------------------------------------------------
    elif self.FunctionType.value == FunctionType.knn.value :

        X = data[ Parameters.columns ]

        if len( Parameters.columns ) > 1 :
            # Remove self from X
            X  = X.drop( self.name, axis = 'columns' )

        X = X.to_numpy()
        y = data[ Parameters.target ].to_numpy()

        # JP: n_neighbors should be a Parameter
        knn = self.Function( n_neighbors = 10, weights = 'distance' )
        knn.fit( X, y )

        nextX = self.FindNextX( X )
        pred  = knn.predict( nextX )
        val   = pred[ 0 ]

    if self.args.DEBUG :
        print( self.name, "val:", val )
        print( '<----- Node:Generate() : ',self.FunctionType.name,self.name )
        print('', flush = True)

    return val

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def FindNextX( self, X ):
    '''Find closest X not equal to X[-1,:], Tp step ahead '''

    x    = X[-1,:]
    Dmin = 1E15
    imin = None

    # Limit X to the data "library" as in EDM
    for i in range( self.Parameters.predictionStart ) :
        xi = X[i,:]
        D  = dist( x, xi )
        if D < Dmin :
            Dmin = D
            imin = i

    # Tp steps ahead
    #if imin + self.Parameters.Tp < X.shape[ 0 ] :
    #    imin = imin + self.Parameters.Tp

    return( X[imin + 1,:].reshape(1,-1) )
