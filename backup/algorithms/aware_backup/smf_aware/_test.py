'''
Created on 06.09.2017

@author: ludewig
'''

import numpy as np

if __name__ == '__main__':
    
    pred = np.array( [0.3,0.1,0.2,0.4,0.5] )
    
    print( np.diag( pred ) )
    print( pred.T )
    
    print( np.diag( pred ) -  pred.T )
    