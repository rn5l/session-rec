import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# data config (all methods)
DATA_PATH = '../data/diginetica/raw/'
DATA_PATH_PROCESSED = '../data/diginetica/prepared_test/'
# DATA_FILE = 'yoochoose-clicks-10M'
# DATA_FILE = 'train-clicks'
# MAP_FILE = 'train-queries'
# MAP_FILE2 = 'train-item-views'
DATA_FILE = 'train-item-views'

# COLS=[0,1,2]
COLS = [0, 2, 3, 4]
# TYPE = 3
TYPE = 2

# filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

# min date config
MIN_DATE = '2016-05-07'

# days test default config
DAYS_TEST = 7

# slicing default config
NUM_SLICES = 5
DAYS_OFFSET = 45
DAYS_SHIFT = 18
DAYS_TRAIN = 25
DAYS_TEST = 7

# retraining default config
DAYS_RETRAIN = 1


# preprocessing from original gru4rec -  uses just the last day as test
def preprocess_org(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                   min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    split_data_org(data, path_proc + file)


# preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                            min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                            min_date=MIN_DATE, days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_min_date(data, min_date)
    data = filter_data(data, min_item_support, min_session_length)
    split_data(data, path_proc + file, days_test)


# preprocessing adapted from original gru4rec
def preprocess_days_test(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                         min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)

    split_data(data, path_proc + file, days_test)


# preprocessing to create data slices with a window
def preprocess_slices(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                      min_session_length=MIN_SESSION_LENGTH,
                      num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN,
                      days_test=DAYS_TEST):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    slice_data(data, path_proc + file, num_slices, days_offset, days_shift, days_train, days_test)


# just load and show info
def preprocess_info(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)


def preprocess_save(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH):
    data = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length)
    data.to_csv(path_proc + file + '_preprocessed.txt', sep='\t', index=False)


# just load and show info
def preprocess_buys(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED):
    data = load_data(path + file)
    data.to_csv(path_proc + file + '.txt', sep='\t', index=False)


def load_data(file):
    if TYPE is 1:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int64, 1: np.int64, 2: np.int64})
        mapping = pd.read_csv(DATA_PATH + MAP_FILE + '.csv', sep=';', usecols=[0, 1, 4, 5],
                              dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        mapping2 = pd.read_csv(DATA_PATH + MAP_FILE2 + '.csv', sep=';', usecols=[0, 2, 3, 4],
                               dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        # specify header names
        data.columns = ['QueryId', 'Time', 'ItemId']
        mapping.columns = ['QueryId', 'SessionId', 'Time2', 'Date']
        # mapping2.columns = ['SessionId', 'ItemId','Time3','Date2']

        data = data.merge(mapping, on='QueryId', how='inner')
        # data = data.merge( mapping2, on=['SessionId','ItemId'], how='outer' )
        del mapping
        # del mapping2
        data.to_csv(file + '.1.csv', index=False)

        # convert time string to timestamp and remove the original column
        #         start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        #         data['Time'] = (data['Time'] / 1000) + start.timestamp()
        #         data['TimeO'] = data.Time.apply( lambda x: datetime.fromtimestamp( x, timezone.utc ) )
        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['TimeNew'] = (data['Time2'] / 1000) + data['Datestamp'] + (data['Time'] / 1000000)
        data['TimeO'] = data.TimeNew.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))
        data.to_csv(file + '.2.csv', index=False)

        data['Time'] = data['TimeNew']
        data.sort_values(['SessionId', 'TimeNew'], inplace=True)
        print(data)


    elif TYPE is 2:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int32, 1: np.int64, 2: np.int64, 3: str})
        # specify header names

        # data.columns = ['SessionId', 'Time', 'ItemId','Date']
        data.columns = ['SessionId', 'ItemId', 'Time', 'Date']
        data = data[['SessionId', 'Time', 'ItemId', 'Date']]
        print(data)
        data['Time'] = data.Time.fillna(0).astype(np.int64)
        # convert time string to timestamp and remove the original column
        # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['Time'] = (data['Time'] / 1000)
        data['Time'] = data['Time'] + data['Datestamp']
        data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))
    
    elif TYPE is 4:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, header=0, dtype={0: np.int32, 1: np.int64, 2: np.int32, 3: str})
        # specify header names
        # data.columns = ['sessionId', 'TimeStr', 'itemId']
        data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.+
        data['SessionId'] = data['sessionId']
        data['ItemId'] = data['itemId']
        del data['itemId'], data['sessionId'], data['eventdate']
        
        data['TimeAdd'] = 1
        data['TimeAdd'] = data.groupby('SessionId').TimeAdd.cumsum()
        data['Time'] += data['TimeAdd']
        print(data)
        del data['TimeAdd']

    elif TYPE is 3:
        # load csv
        data = pd.read_csv(file + '.csv', sep=';', usecols=COLS, dtype={0: np.int64, 1: np.int64, 2: np.int64})
        mapping = pd.read_csv(DATA_PATH + MAP_FILE + '.csv', sep=';', usecols=[0, 1, 4, 5],
                              dtype={0: np.int64, 1: np.int64, 2: np.int64, 3: str})
        # specify header names
        data.columns = ['QueryId', 'Time', 'ItemId']
        mapping.columns = ['QueryId', 'SessionId', 'Time2', 'Date']

        data = data.merge(mapping, on='QueryId', how='inner')
        # data = data.merge( mapping2, on=['SessionId','ItemId'], how='outer' )
        del mapping

        # convert time string to timestamp and remove the original column
        # start = datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
        # data['Time'] = (data['Time'] / 1000) + start.timestamp()
        # data['TimeO'] = data.Time.apply( lambda x: datetime.fromtimestamp( x, timezone.utc ) )

        data['Date'] = data.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['Datestamp'] = data['Date'].apply(lambda x: x.timestamp())
        data['Time'] = (data['Time'] / 1000000) + data['Datestamp']
        data['TimeO'] = data.Time.apply(lambda x: datetime.fromtimestamp(x, timezone.utc))

        print(data)
        data['SessionId'] = data['QueryId']
        data.sort_values(['QueryId', 'Time'], inplace=True)

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    data = data.groupby('SessionId').apply(lambda x: x.sort_values('Time'))     # data = data.sort_values(['SessionId'],['Time'])
    data.index = data.index.get_level_values(1)
    return data;


def filter_data(data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH):
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    session_lengths = session_lengths[ session_lengths >= min_session_length ]
    data = data[np.in1d(data.SessionId, session_lengths.index)]

    # filter item support
    data['ItemSupport'] = data.groupby('ItemId')['ItemId'].transform('count')
    data = data[data.ItemSupport >= min_item_support]

    # filter session length again, after filtering items
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    
    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set default \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def filter_min_date(data, min_date='2014-04-01'):
    
    print('filter_min_date')
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    # filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[session_max_times > min_datetime.timestamp()].index

    data = data[np.in1d(data.SessionId, session_keep)]

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set min date \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def split_data_org(data, output_file):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times >= tmax - 86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)


def split_data(data, output_file, days_test=DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

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
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)

def slice_data(data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ):
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))

    start = datetime.fromtimestamp(data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(slice_id) + '.txt', sep='\t', index=False)

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(),
                 end.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)


# def retrain_data(data, output_file_path, output_file_name, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
def retrain_data(data, output_file, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
    retrain_num = int(days_test/days_retrain)
    for retrain_n in range(0, retrain_num):
        # output_f = output_file_path + 'set_' + str(retrain_n) + '/' + output_file_name
        # split_data_retrain(data, output_file, days_train, days_retrain, retrain_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)
        train = split_data_retrain_train(data, output_file, days_train, days_retrain, retrain_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)
        test_set_num = retrain_num - retrain_n
        for test_n in range(0,test_set_num):
            split_data_retrain_test(data, train, output_file, days_train, days_retrain, retrain_n, test_n)  #split_data_retrain(data, output_file, days_train, days_test, file_num)


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
    # session_test = session_max_times[(session_max_times > train_to.timestamp()) & (session_max_times <= test_to.timestamp())].index

    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    # test = data[np.in1d(data.SessionId, session_test)]
    # test = test[np.in1d(test.ItemId, train.ItemId)]
    # tslength = test.groupby('SessionId').size()
    # test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), train_from.date().isoformat(),
                 train_to.date().isoformat()))
    train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
    #                                                                    test.ItemId.nunique()))
    # test.to_csv(output_file + '_test.' + str(retrain_num) + '.txt', sep='\t', index=False)

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
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test_from.date().isoformat(),
                 test_to.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(retrain_num) + '_' + str(test_set_num) + '.txt', sep='\t', index=False)

    # data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    # valid_from = data_end - timedelta(days=days_test)
    # session_max_times = train.groupby('SessionId').Time.max()
    # session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    # session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    # train_tr = train[np.in1d(train.SessionId, session_train)]
    # valid = train[np.in1d(train.SessionId, session_valid)]
    # valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    # tslength = valid.groupby('SessionId').size()
    # valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    # print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
    #                                                                     train_tr.ItemId.nunique()))
    # train_tr.to_csv(output_file + '_train_tr.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
    #                                                                          valid.ItemId.nunique()))
    # valid.to_csv(output_file + '_train_valid.' + str(retrain_num) + '.txt', sep='\t', index=False)

# -------------------------------------
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    #preprocess_info()
    preprocess_org_min_date(min_date=MIN_DATE, days_test=DAYS_TEST)
    #preprocess_days_test(days_test=DAYS_TEST)
    #preprocess_slices()
