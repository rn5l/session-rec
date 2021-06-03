import pandas as pd
import numpy as np
from datetime import datetime, timezone

PATH = '../../../data/xing/xing2016/'
FILE = 'interactions'

# keys
USER_KEY='user_id'
ITEM_KEY='item_id'
TIME_KEY='created_at'
SESSION_KEY='session_id'
TYPE_KEY='interaction_type'

SESSION_THRESHOLD = 30 * 60
MIN_ITEM_SUPPORT = 5  # 20
MIN_SESSION_LENGTH = 2  # 3
MIN_USER_SESSIONS = 3  # 5
MAX_USER_SESSIONS = None  # 200
REPEAT = False  # apply filters several times
SAMPLE_NUM = 5

def make_sessions(data, session_th=SESSION_THRESHOLD, is_ordered=False, user_key=USER_KEY, time_key=TIME_KEY, session_key=SESSION_KEY):
    """Assigns session ids to the events in data without grouping keys"""
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
    # drop duplicate interactions within the same session
    # data.drop_duplicates(subset=[ITEM_KEY, SESSION_KEY, TYPE_KEY], keep='first', inplace=True)
    condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
        [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
        [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
    count = 1
    while not condition:
        print(count)
        # keep items with >=20 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= MIN_ITEM_SUPPORT].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 3
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= MIN_SESSION_LENGTH].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        if MAX_USER_SESSIONS is None:  # no filter for max number of sessions for each user
            good_users = sess_per_user[(sess_per_user >= MIN_USER_SESSIONS)].index
        else:
            good_users = sess_per_user[(sess_per_user >= MIN_USER_SESSIONS) & (sess_per_user < MAX_USER_SESSIONS)].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
            [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
            [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
        count += 1
        if not REPEAT:
            break

    # output
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
    data = pd.read_csv(PATH + FILE + '.csv', sep='\t')
    # remove interactions of type 'delete'
    data = data[data[TYPE_KEY] != 4].copy()

    # partition interactions into sessions with 30-minutes idle time
    data = make_sessions(data)

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