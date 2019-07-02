import preprocessing.preprocess_rsc15 as pp
import time

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a window approach  
    buys: load buys and safe file to prepared
'''
METHOD = "org_min_date"

'''
data config (all methods)
'''
PATH = 'data/rsc15/raw/'
PATH_PROCESSED = 'data/rsc15/prepared/'
FILE = 'rsc15-clicks'

'''
org_min_date config
'''
MIN_DATE = '2014-07-01'

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

'''
days test default config
'''
DAYS_TEST = 1

'''
slicing default config
'''
NUM_SLICES = 5 #offset in days from the first date in the data set
DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
DAYS_SHIFT = 31
#each slice consists of...
DAYS_TRAIN = 30
DAYS_TEST = 1

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    print( "START preprocessing ", METHOD )
    sc, st = time.clock(), time.time()
    
    if METHOD == "info":
        pp.preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
    
    elif METHOD == "org":
        pp.preprocess_org( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
     
    elif METHOD == "org_min_date":
        pp.preprocess_org_min_date( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MIN_DATE )
        
    elif METHOD == "day_test":
        pp.preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_TEST )
    
    elif METHOD == "slice":
        pp.preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
        
    elif METHOD == "buys":
        pp.preprocess_buys( PATH, FILE, PATH_PROCESSED )
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preproccessing ", (time.clock() - sc), "c ", (time.time() - st), "s" )
    