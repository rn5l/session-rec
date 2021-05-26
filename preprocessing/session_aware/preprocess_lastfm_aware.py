import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import dateutil.parser
import subprocess

# data config (all methods)

# file params
FILE_TYPE_PREFIX = '.hdf'
CLEAN_TEST = True # Preprocess the test set

# keys
USER_KEY='UserId'
ITEM_KEY='ItemId'
TIME_KEY='Time'
SESSION_KEY='SessionId'

SESSION_THRESHOLD = 30 * 60
# filtering config (all methods)
MIN_ITEM_SUPPORT = 5
MIN_SESSION_LENGTH = 2
MIN_USER_SESSIONS = 3
MAX_SESSION_LENGTH = 20  # 40
# MAX_USER_SESSIONS = 200
SLICES_NUM = 5
SLICE_INTERVAL = 217 # total_interval = 1587
DAYS_OFFSET = 500  # skip first 1/3 of data because it contains much fewer interactions (and users) in comparision with last 2/3

def prepare_time(data, time_key=TIME_KEY):
    print("prepare_time")
    """Assigns session ids to the events in data without grouping keys"""
    # timestamp = (dateutil.parser.parse(data[ITEM_KEY])).timestamp()
    data[time_key] = data[time_key].apply(lambda x: (dateutil.parser.parse(x)).timestamp())
    return data

def map_user_and_item_id(data):
    print("map_user_and_item_id")
    item_map = {}
    user_map = {}
    for index, row in data.iterrows():
        user_id = row[USER_KEY]
        artist_id = row[ITEM_KEY]

        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if artist_id not in item_map:
            item_map[artist_id] = len(item_map)

        data.at[index, USER_KEY] = user_map[user_id]
        data.at[index, ITEM_KEY] = item_map[artist_id]

    return data

def make_sessions(data, session_th=SESSION_THRESHOLD, is_ordered=False, user_key=USER_KEY, time_key=TIME_KEY, session_key=SESSION_KEY):
    """Assigns session ids to the events in data without grouping keys"""
    print("make_sessions")
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:] != data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data[session_key] = session_ids
    return data

def load_data(file):
    data = pd.read_csv(file + '.tsv', sep='\t',
                       names=[USER_KEY, TIME_KEY, "artist_id", "artist", ITEM_KEY, "track"])  # Items are tracks
    # data = pd.read_csv(file + '.tsv', sep='\t', names=[USER_KEY, TIME_KEY, ITEM_KEY, "artist", "track_id", "track"])  # Items are artists

    # just keep columns USER_KEY, TIME_KEY and ITEM_KEY
    data = data[[USER_KEY, TIME_KEY, ITEM_KEY]]

    # remove rows with NA userId
    data = data[~pd.isnull(data[ITEM_KEY])].copy()

    # prepare time format
    data = prepare_time(data, time_key=TIME_KEY)

    # Map ids
    data = map_user_and_item_id(data)

    # partition interactions into sessions with 30-minutes idle time
    data = make_sessions(data)

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Original data:')
    print('Num Events: {}'.format(len(data)))
    print('Num items: {}'.format(data[ITEM_KEY].nunique()))
    print('Num users: {}'.format(data[USER_KEY].nunique()))
    print('Num sessions: {}'.format(data[SESSION_KEY].nunique()))
    print('Span: {} / {}'.format(data_start.date().isoformat(), data_end.date().isoformat()))
    return data;

def filter_data(data, min_item_support= MIN_ITEM_SUPPORT, min_session_length= MIN_SESSION_LENGTH, min_user_sessions=MIN_USER_SESSIONS, max_session_length= MAX_SESSION_LENGTH):
    condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= min_user_sessions and data.groupby(
        [USER_KEY, SESSION_KEY]).size().min() >= min_session_length and data.groupby([ITEM_KEY]).size().min() >= min_item_support and data.groupby(
        [USER_KEY, SESSION_KEY]).size().max() <= max_session_length
    count = 0
    while not condition:
        # keep items with >=5 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= min_item_support].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 2
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= min_session_length].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # remove sessions with length > 40
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length <= max_session_length].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 2 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        good_users = sess_per_user[sess_per_user >= min_user_sessions].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= min_user_sessions and data.groupby(
            [USER_KEY, SESSION_KEY]).size().min() >= min_session_length and data.groupby([ITEM_KEY]).size().min() >= min_item_support and data.groupby(
            [USER_KEY, SESSION_KEY]).size().max() <= max_session_length
        # condition = false #if want to apply the filters once
        count += 1
        print(count)

    # output
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Filter applied '+str(count)+' times!')
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

def split_data(data, output_file, min_session_length, slice_id= None): #TODO: extend for supproting more than one sessions per use for test
    """
        assign the last session of every user to the test set and the remaining ones to the training set
        """
    train_full_sessions, test_sessions = last_session_out_split(data, min_session_length)
    train_valid_sessions, valid_sessions = last_session_out_split(train_full_sessions, min_session_length)

    data = train_full_sessions
    print('Training data \n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique()))

    data = test_sessions
    print('Test data \n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique()))

    data = train_valid_sessions
    print('Validation_training data \n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique()))

    data = valid_sessions
    print('Validation_test data \n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique()))

    print('Write to disk')
    # write to disk
    # subprocess.call(['mkdir', '-p', 'dense/last-session-out'])
    if slice_id is not None:
        output_file = output_file +"."+ str(slice_id)
    output_file = output_file + FILE_TYPE_PREFIX
    subprocess.call(['mkdir', '-p', 'prepared'])
    train_full_sessions.to_hdf(output_file, 'train')
    test_sessions.to_hdf(output_file, 'test')
    train_valid_sessions.to_hdf(output_file, 'valid_train')
    valid_sessions.to_hdf(output_file, 'valid_test')

def slice_data(data, output_file, num_slices=SLICES_NUM, days_offset=DAYS_OFFSET, days_shift=SLICE_INTERVAL, min_session_length=MIN_SESSION_LENGTH):
    for slice_id in range(0, num_slices):
        print("slice " + str(slice_id))
        split_data_slice(data, output_file, slice_id, days_offset, days_shift, min_session_length)

def split_data_slice(data, output_file, slice_id, days_offset, days_shift, min_session_length):
    start_day = days_offset + (slice_id * days_shift)
    start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(start_day)
    end_day = days_offset + ((slice_id+1) * days_shift)
    end = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(end_day)

    # prefilter the timespan
    session_max_times = (data.groupby([SESSION_KEY])[TIME_KEY]).max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data = data[np.in1d(data[SESSION_KEY], greater_start.intersection(lower_end))]

    print("filter data")
    data = filter_data(data)

    print("split data")
    # training-test split
    split_data(data, output_file, min_session_length, slice_id)