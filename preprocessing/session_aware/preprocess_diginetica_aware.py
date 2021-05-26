import numpy as np
import pandas as pd
from datetime import datetime, timezone
import subprocess

# data config (all methods)

# file params
FILE_TYPE_PREFIX = '.hdf'
CLEAN_TEST = True # Preprocess the test set

# keys
USER_KEY='userId'
ITEM_KEY='itemId'
TIME_KEY='eventdate'
SESSION_KEY='sessionId'


# filtering config (all methods)
MIN_ITEM_SUPPORT = 5
MIN_SESSION_LENGTH = 2
MIN_USER_SESSIONS = 3



def clear_sessions(data):
    """Delete sessions which the session_id is the same for different users!"""
    data = data[data[SESSION_KEY].isin(data.groupby(SESSION_KEY)[USER_KEY].nunique()[
                                           (data.groupby(SESSION_KEY)[USER_KEY].nunique() > 1) == False].index)]
    return data

def prepare_time(data, time_key=TIME_KEY):
    """Assigns session ids to the events in data without grouping keys"""
    data[time_key] = data[time_key].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    data['tmp'] = 1
    data['tmp'] = data.groupby(SESSION_KEY).tmp.cumsum()
    data[time_key] = data[time_key] + data['tmp']
    del data['tmp']
    return data

def load_data(file):
    data = pd.read_csv(file + '.csv', header=0, sep=';')
    print('Cleaning data')
    # remove rows with NA userId
    data = data[~np.isnan(data[USER_KEY])].copy()
    # to delete sessions which the session_id is the same for different users!
    data = clear_sessions(data)
    # prepare time format
    data = prepare_time(data, time_key=TIME_KEY)

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Original data:')
    print('Num Events: {}'.format(len(data)))
    print('Num items: {}'.format(data[ITEM_KEY].nunique()))
    print('Num users: {}'.format(data[USER_KEY].nunique()))
    print('Num sessions: {}'.format(data[SESSION_KEY].nunique()))
    print('Span: {} / {}'.format(data_start.date().isoformat(), data_end.date().isoformat()))
    return data;

def filter_data(data, min_item_support= MIN_ITEM_SUPPORT, min_session_length= MIN_SESSION_LENGTH, min_user_sessions=MIN_USER_SESSIONS):
    # keep items with >=5 interactions
    item_pop = data[ITEM_KEY].value_counts()
    good_items = item_pop[item_pop >= min_item_support].index
    data = data[data[ITEM_KEY].isin(good_items)]
    # remove sessions with length < 2
    session_length = data[SESSION_KEY].value_counts()
    good_sessions = session_length[session_length >= min_session_length].index
    data = data[data[SESSION_KEY].isin(good_sessions)]
    # let's keep only returning users (with >= 2 sessions)
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    good_users = sess_per_user[sess_per_user >= min_user_sessions].index
    data = data[data[USER_KEY].isin(good_users)]

    # output
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Filtered data:')
    print('Num Events: {}'.format(len(data)))
    print('Num items: {}'.format(data[ITEM_KEY].nunique()))
    print('Num users: {}'.format(data[USER_KEY].nunique()))
    print('Num sessions: {}'.format(data[SESSION_KEY].nunique()))
    print('Span: {} / {}'.format(data_start.date().isoformat(), data_end.date().isoformat()))

    return data

def last_session_out_split(data, min_session_length):
    """
    last-session-out split
    assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[USER_KEY, TIME_KEY]).groupby(USER_KEY)[SESSION_KEY]
    last_session = sessions.last()
    train = data[~data[SESSION_KEY].isin(last_session.values)].copy()
    test = data[data[SESSION_KEY].isin(last_session.values)].copy()
    if CLEAN_TEST:
        train_items = train[ITEM_KEY].unique()
        test = test[test[ITEM_KEY].isin(train_items)]
        #  remove sessions in test shorter than min_session_length
        slen = test[SESSION_KEY].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test[test[SESSION_KEY].isin(good_sessions)].copy()
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
    return train, test

def split_data(data, file, min_session_length, test_sessions): #TODO: extend for supproting more than one sessions per use for test
    """
        assign the last session of every user to the test set and the remaining ones to the training set
        """
    train_full_sessions, test_sessions = last_session_out_split(data, min_session_length)
    train_valid_sessions, valid_sessions = last_session_out_split(train_full_sessions, min_session_length)

    print('Training data:')
    print('Num Events: {}'.format(len(train_full_sessions)))
    print('Num items: {}'.format(train_full_sessions[ITEM_KEY].nunique()))
    print('Num users: {}'.format(train_full_sessions[USER_KEY].nunique()))
    print('Num sessions: {}'.format(train_full_sessions[SESSION_KEY].nunique()))

    print('Test data:')
    print('Num Events: {}'.format(len(test_sessions)))
    print('Num items: {}'.format(test_sessions[ITEM_KEY].nunique()))
    print('Num users: {}'.format(test_sessions[USER_KEY].nunique()))
    print('Num sessions: {}'.format(test_sessions[SESSION_KEY].nunique()))

    print('Validation_training data:')
    print('Num Events: {}'.format(len(train_valid_sessions)))
    print('Num items: {}'.format(train_valid_sessions[ITEM_KEY].nunique()))
    print('Num users: {}'.format(train_valid_sessions[USER_KEY].nunique()))
    print('Num sessions: {}'.format(train_valid_sessions[SESSION_KEY].nunique()))

    print('Validation_test data:')
    print('Num Events: {}'.format(len(valid_sessions)))
    print('Num items: {}'.format(valid_sessions[ITEM_KEY].nunique()))
    print('Num users: {}'.format(valid_sessions[USER_KEY].nunique()))
    print('Num sessions: {}'.format(valid_sessions[SESSION_KEY].nunique()))

    print('Write to disk')
    # write to disk
    # subprocess.call(['mkdir', '-p', 'dense/last-session-out'])
    file = file + FILE_TYPE_PREFIX
    subprocess.call(['mkdir', '-p', 'prepared'])
    train_full_sessions.to_hdf(file, 'train')
    test_sessions.to_hdf(file, 'test')
    train_valid_sessions.to_hdf(file, 'valid_train')
    valid_sessions.to_hdf(file, 'valid_test')