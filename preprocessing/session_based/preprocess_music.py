import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
import random
import time
from theano.tensor import inplace

#data config (all methods)
DATA_PATH = '../data/playlist/raw/'
DATA_PATH_PROCESSED = '../data/playlist/prepared/'
#DATA_FILE = 'yoochoose-clicks-10M'
DATA_FILE = '30music-50k-sessions'

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 2

#days test default config 
#DAYS_TEST = 90

#slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 0
DAYS_SHIFT = 95
DAYS_TRAIN = 90
DAYS_TEST = 5

# retraining default config
DAYS_RETRAIN = 1


#preprocessing from original gru4rec
def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
#    data = load_data( path+file )
    #for listening logs
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )
    
#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
#    data = load_data( path+file )
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )      
    
    
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep='\t' )
    #user_id,session_id,track_id,time_stamp,artist_id
    
    #data.sort_values( by=['Time'], inplace=True )
    #data['SessionId'] = data.groupby( [data.SessionId] ).grouper.group_info[0]
    
    data.sort_values( by=['SessionId','Time'], inplace=True )
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    
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
    test_from = data_end - timedelta( days_test )
    
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
    
    
    
def slice_data( data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ): 
    
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


def retrain_data(data, output_file, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
    retrain_num = int(days_test/days_retrain)
    for retrain_n in range(0, retrain_num):
        train = split_data_retrain_train(data, output_file, days_train, days_retrain, retrain_n)
        test_set_num = retrain_num - retrain_n
        for test_n in range(0,test_set_num):
            split_data_retrain_test(data, train, output_file, days_train, days_retrain, retrain_n, test_n)

def split_data_retrain_train(data, output_file, days_train, days_test, retrain_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    train_from = data_start
    new_days = retrain_num * days_test
    train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    # todo: test_from
    # test_to = train_to + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index

    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    # print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
    #                                                                          train.ItemId.nunique()))

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), train_from.date().isoformat(),
                 train_to.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    valid_from = data_end - timedelta(days=days_test)
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.' + str(retrain_num) + '.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.' + str(retrain_num) + '.txt', sep='\t', index=False)

    return train


def split_data_retrain_test(data, train, output_file, days_train, days_test, retrain_num, test_set_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    # train_from = data_start
    # new_days = retrain_num * days_test
    # new_days = test_set_num * days_test
    new_days = (retrain_num + test_set_num) * days_test
    # train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_from = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_to = test_from + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    # session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index
    # session_test = session_max_times[(session_max_times > train_to.timestamp()) & (session_max_times <= test_to.timestamp())].index
    session_test = session_max_times[(session_max_times > test_from.timestamp()) & (session_max_times <= test_to.timestamp())].index

    # train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    # print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
    #                                                                          train.ItemId.nunique()))
    # train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
    #                                                                    test.ItemId.nunique()))

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test_from.date().isoformat(),
                 test_to.date().isoformat()))

    if(test.empty): #todo: or handle it while reading the data in running experiments
        print('Test data is empty!!!')
    else:
        test.to_csv(output_file + '_test.' + str(retrain_num) + '_' + str(test_set_num) + '.txt', sep='\t', index=False)



# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    
#    preprocess_info();
#    preprocess_org();
    preprocess_slices();
