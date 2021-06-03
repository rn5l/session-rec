import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
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
PATH = '../../../../data/lastfm/'
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
SLICES_NUM = 5
# start: 2005-02-14
# end: 2009-06-19
# outliers: 2010-08-13 & 2013-09-29
SLICE_INTERVAL = 217 # total_interval = 1587
DAYS_OFFSET = 500 # 2
# SLICE_INTERVAL = 227 # 267 # 277 # 287 #307# 317  # total_interval = 1587
# DAYS_OFFSET = 450 # 250 # 200 # 150 #100 #50 # 0

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

def slice_data(data, num_slices=SLICES_NUM, days_offset=DAYS_OFFSET, days_shift=SLICE_INTERVAL):
    for slice_id in range(0, num_slices):
        split_data_slice(data, slice_id, days_offset, days_shift)


def split_data_slice(data, slice_id, days_offset, days_shift):

    start_day = days_offset + (slice_id * days_shift)
    start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(start_day)
    end_day = days_offset + ((slice_id + 1) * days_shift)
    end = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(end_day)
    session_max_times = (data.groupby([SESSION_KEY])[TIME_KEY]).max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data = data[np.in1d(data[SESSION_KEY], greater_start.intersection(lower_end))]

    # print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
    #       format(slice_id, len(data_filtered), data_filtered[SESSION_KEY].nunique(), data_filtered[ITEM_KEY].nunique(),
    #              start.date().isoformat(), end.date().isoformat()))

    print('Slice data set {}\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(slice_id, len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 start.date().isoformat(), end.date().isoformat()))

    print('--------------------- Slice-Original---')
    report_statistics(data)

    filter_data(data)
    #todo: training-test split


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

    print('--------------------- Slice-Filtered---')
    report_statistics(data)

    return data


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

    data = pd.read_csv(PATH + FILE + '_prepared.txt', sep='\t')  # Items are tracks

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

    slice_data(data)