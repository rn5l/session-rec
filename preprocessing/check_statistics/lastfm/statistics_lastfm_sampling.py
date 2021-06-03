import pandas as pd
import numpy as np
from datetime import datetime, timezone
import dateutil.parser

# EC
# diginetica:
    # - train-item-views (some entries (NA))
    # - train-purchases (some entries (NA))

# retailrocket - events

# tmall [Private]
    # - dataset15
    # - lso_train

# Xing 2016 - interactions
PATH = '../../../data/lastfm/'
FILE = 'userid-timestamp-artid-artname-traid-traname'
# FILE = 'lastfm_sample'
# keys
USER_KEY='user_id'
ITEM_KEY='item_id'
TIME_KEY='created_at'
SESSION_KEY='session_id'
TYPE_KEY='interaction_type'
# filters
SESSION_THRESHOLD = 30 * 60
MIN_ITEM_SUPPORT = 5
MIN_SESSION_LENGTH = 2
MIN_USER_SESSIONS = 3
MAX_SESSION_LENGTH = 20
# MAX_USER_SESSIONS = 200
REPEAT = False  # apply filters several times
SAMPLE_NUM = 5

# Xing 2017 - interactions

# zalando - lso_train [private]


# Music
# 8tracks - Lso_train / lso_test [private]
# 30music - 30music-200ks
# aotm: [Playlists were randomly distributed to a time span of one year]
    # - playlists-aotm
    # - playlists-aotm_asyear
# lastfm [don’t have raw data]
# nowplaying - nowplaying


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


def filter_data(data):
    condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
        [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
        [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT and data.groupby(
        [USER_KEY, SESSION_KEY]).size().max() <= MAX_SESSION_LENGTH
    count = 1
    while not condition:
        print(count)
        # keep items with >=5 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= MIN_ITEM_SUPPORT].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 2
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= MIN_SESSION_LENGTH].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # remove sessions with length > 40
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length <= MAX_SESSION_LENGTH].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 3 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        good_users = sess_per_user[sess_per_user >= MIN_USER_SESSIONS].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
            [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
            [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT and data.groupby(
            [USER_KEY, SESSION_KEY]).size().max() <= MAX_SESSION_LENGTH
        count += 1
        if not REPEAT:
            break

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Filtered---')
    report_statistics(data)


def report_statistics(data):
    print('--------------------- Statistics---')
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    print('---------------------')
    # print('Num of users: {}'.format(data[USER_KEY].nunique()))
    # print('Max num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().max()))
    # print('Min num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().min()))
    # print('Median num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().median()))
    # print('Mean num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().mean()))
    # print('Std num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().std()))
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Max num of users\' sessions: {}'.format(sess_per_user.max()))
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Median num of users\' sessions: {}'.format(sess_per_user.median()))
    print('Mean num of users\' sessions: {}'.format(sess_per_user.mean()))
    print('Std num of users\' sessions: {}'.format(sess_per_user.std()))
    print('---------------------')
    # print('Num of sessions per user: {}'.format(np.count_nonzero(data.groupby(USER_KEY)[SESSION_KEY].nunique())))
    print('Max sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().max()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Median sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().median()))
    print('Mean sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().mean()))
    print('Std sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().std()))
    print('---------------------')
    # print('Num of items: {}'.format(data[ITEM_KEY].nunique()))
    # print('Max num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().max()))
    # print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    # print('Median num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().median()))
    # print('Mean num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().mean()))
    # print('Std num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().std()))

    print('Max num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().max()))
    print('Min num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().min()))
    print('Median num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().median()))
    print('Mean num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().mean()))
    print('Std num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().std()))
    print('---------------------')


if __name__ == '__main__':

    # updater.dispatcher.add_handler( CommandHandler('status', status) )
    data = pd.read_csv(PATH + FILE + '.tsv', sep='\t', names=[USER_KEY, TIME_KEY, "artist_id", "artist", ITEM_KEY, "track"])   # Items are tracks
    # data = pd.read_csv(PATH + FILE + '.tsv', sep='\t', names=[USER_KEY, TIME_KEY, "artist_id", "artist", ITEM_KEY, "track"])  # Items are artists
    # just keep columns USER_KEY, TIME_KEY and ITEM_KEY
    data = data[[USER_KEY, TIME_KEY, ITEM_KEY]]
    # remove rows with NA item_id
    data = data[~pd.isnull(data[ITEM_KEY])].copy()
    # TODO: appropriate preprocessing for data[TIME_KEY]

    # prepare time format
    data = prepare_time(data, time_key=TIME_KEY)

    data = map_user_and_item_id(data)

    # partition interactions into sessions with 30-minutes idle time
    data = make_sessions(data)

    # to delete sessions which the session_id is the same for different users!
    # data = data[data[SESSION_KEY].isin(data.groupby(SESSION_KEY)[USER_KEY].nunique()[
    #                                        (data.groupby(SESSION_KEY)[USER_KEY].nunique() > 1) == False].index)]

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Original data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Original---')
    report_statistics(data)

    users = data.groupby([USER_KEY]).size().index
    users_df = pd.DataFrame(users)
    # shuffle users and partition them into 5 parts
    from sklearn.utils import shuffle

    users_df = shuffle(users_df)

    user_samples_list = np.array_split(users_df, SAMPLE_NUM)
    # (user_samples_list[0].index).intersection(user_samples_list[1].index)
    for idx in range(0, len(user_samples_list)):
        user_samples = data[data[USER_KEY].isin(user_samples_list[idx].index)]
        user_samples.reset_index(inplace=True, drop=True)
        data_start = datetime.fromtimestamp(user_samples[TIME_KEY].min(), timezone.utc)
        data_end = datetime.fromtimestamp(user_samples[TIME_KEY].max(), timezone.utc)

        print('Sample data set {}\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
              format(idx, len(user_samples), user_samples[USER_KEY].nunique(), user_samples[SESSION_KEY].nunique(),
                     user_samples[ITEM_KEY].nunique(),
                     data_start.date().isoformat(), data_end.date().isoformat()))
        print('--------------------- Sample-Original---')
        report_statistics(user_samples)

        filter_data(user_samples)

