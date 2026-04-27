# Python distribution modules
# Community modules
from scipy.stats import kstest
# Local modules

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def KStest( x, y, xName = '', yName = '', verbose = False ):
    '''
    x, y : observed variables
    
    Return KstestResult object :
      statistic          - KS test statistic
      pvalue             - p-value to accept null (p < α : reject null)
      statistic_location - distance is measured at this observation
      statistic_sign     - +1 if cdf(x) > cdf(y) @statistic_location, else -1

    If p-value <= α null hypothesis is rejected
    if p-value > α, null hypothesis is not rejected.

    A small p-value (typically < 0.05) rejects the null hypothesis (H0)
    that the samples come from the same distribution suggesting they are
    different. A large p-value (>= 0.05) fails to reject H0, indicating
    insufficient evidence to say they're different. The p-value answers:
    "If the populations were identical, what's the chance of seeing data
    this different?"
    '''
    KS = kstest( x, # 1-D array of observations
                 y, # 1-D array of observations -> two sided test ks_2samp()
                 alternative = 'two-sided', # null: distributions are identical
                 nan_policy = 'propagate',
                 keepdims = False )

    if verbose :
        print( f'KS({xName}:{yName}) {KS.statistic.round(4)}' +\
               f' p-value {KS.pvalue.round(4)}')

    return KS
