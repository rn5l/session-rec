import numpy as np
import pandas as pd
from _datetime import timezone, datetime, timedelta

#data config (all methods)
DATA_PATH = '../../dressipy/store.h5'
DATA_PATH_PROCESSED = '../data/dressipi/preparedDS/'
#DATA_FILE = 'yoochoose-clicks-10M'
DATA_FILE = 'views'
DATA_FILE_BUYS = 'transactions'
SESSION_LENGTH = 30 * 60 #30 minutes


#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'

#days test default config 
DAYS_TEST = 4

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
#DAYS_TEST = 1


#preprocessing from original gru4rec
def preprocess_org( path=DATA_PATH, file=DATA_FILE, buys_file=DATA_FILE_BUYS, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path, file, buys_file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )
    buys.to_csv( path_proc + buys_file + '.txt', sep='\t', index=False)

#preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date( path=DATA_PATH, file=DATA_FILE, buys_file=DATA_FILE_BUYS, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, min_date=MIN_DATE ):
    
    data, buys = load_data( path, file, buys_file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data_org( data, path_proc+file )
    buys.to_csv( path_proc + buys_file + '.txt', sep='\t', index=False)
    
#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, buys_file=DATA_FILE_BUYS, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
    data, buys = load_data( path, file, buys_file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )
    print(buys)
    buys.to_csv( path_proc + buys_file + '.txt', sep='\t', index=False)

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, buys_file=DATA_FILE_BUYS, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data, buys = load_data( path, file, buys_file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    buys.to_csv( path_proc + buys_file + '.txt', sep='\t', index=False)
    
#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, buys_file=DATA_FILE_BUYS, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path, file, buys_file )
    data = filter_data( data, min_item_support, min_session_length )  
    
def load_data( store, name, name_buys ) : 
     
    store = pd.HDFStore( store )
    data = store[name]
    buys = store[name_buys]
    
    del( data['Time'] )
     
    data['SessionId'] = data['SessionDay']
    data['ItemId'] = data['Item']
    data['Time'] = data['TimeObject'].apply( lambda t: t.timestamp() )
    data['UserId'] = data['User']
    data['TimeO'] = data['TimeObject']
    del( data['Session'] )
    del( data['SessionDay'] )
    del( data['Item'] )
    del( data['User'] )
    del( data['TimeObject'] )
    
    data.sort_values( ['SessionId','Time'], inplace=True )
        
    sessionid_map = {} # day => user => id
    sessiontime_map = {} # session => time
    
    for row in data.itertuples(index=False): 
        
        user, session, time = row[3], row[0], row[4]
        
        key = time.date()
        
        if not key in sessionid_map:
            sessionid_map[key] = {}
            sessiontime_map[key] = {}
            
        if not user in sessionid_map[key]:
            sessionid_map[key][user] = {}
            sessiontime_map[key][user] = {}
            
        sessionid_map[key][user] = session
        sessiontime_map[session] = time.timestamp()
                
    del( data['TimeO'] )
     
    buys['SessionId'] = buys.apply( lambda row: sessionid_map[row['Day'].date()][row['User']] if row['Day'].date() in sessionid_map and row['User'] in sessionid_map[row['Day'].date()] else -1, axis=1  )
    buys['ItemId'] =  buys['Item']
    
    buys['TimeO'] = buys['Day']
    del(buys['Time'])
    buys = buys[ buys['SessionId'] > 0 ]
    buys['Time'] = buys.apply( lambda row: sessiontime_map[row['SessionId']] + 1, axis=1  )
    
    buys['UserId'] =  buys['User']
    del(buys['User'])
    del(buys['Item'])
    del(buys['Day'])
    del(buys['TimeO'])
    
    
    
    return data, buys;

def filter_data( data, min_item_support, min_session_length ) : 
        
    #y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;

def filter_min_date( data, min_date='2014-04-01' ) :
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    #filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[ session_max_times > min_datetime.timestamp() ].index
    
    data = data[ np.in1d(data.SessionId, session_keep) ]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;

def split_data_org( data, output_file ) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)
    
    
    
def split_data( data, output_file, days_test ) :
    
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days=days_test )
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    
    
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc ) + timedelta( days_offset ) 
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)


# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    
    preprocess_info();
