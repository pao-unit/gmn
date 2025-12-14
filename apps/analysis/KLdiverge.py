# Python distribution modules
# Community modules
from scipy.special import rel_entr # scipy relative entropy
#from scipy.stats import entropy   # alternative entropy(p, q, nan_policy)
# Local modules

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def KLdiverge( p, q, pName = '', qName = '', verbose = False ):
    '''Kullback–Leibler divergence of p | q
       Return KL divergence
    '''
    KL = rel_entr(p,q).sum()

    if verbose :
        print( f'KL({pName}||{qName}) {KL.round(4)}' )

    return KL
