import numpy as np
import pandas as pd
from math import log10
import collections as col
from datetime import datetime as dt
from datetime import timedelta as td
from algorithms.extensions.reminder import Reminder

class USequentialRules:
    '''
    Code based on work by Kamehkhosh et al.,A Comparison of Frequent Pattern Techniques and a Deep Learning Method for Session-Based Recommendation, TempRec Workshop at ACM RecSys 2017.

    SequentialRules(steps = 3, weighting='div', pruning=0.0)

    Parameters
    --------
    steps : int
        TODO. (Default value: 3)
    weighting : string
        TODO. (Default value: 3)
    pruning : float
        TODO. (Default value: 0)

    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    user_key : string
        Header of the user ID column in the input file. (default: 'UserId')

    boost_own_sessions: double
        to increase the impact of (give weight more weight to) the sessions which belong to the user. (default: None)
        the value will be added to 1.0. For example for boost_own_sessions=0.2, weight will be 1.2

    reminders: bool
        Include reminding items in the (main) recommendation list. (default: False)

    remind_strategy: string
        Ranking strategy of the reminding list (default: recency)

    remind_sessions_num: int
        Number of the last user's sessions that the possible items for reminding are taken from (default: 6)

    reminders_num: int
        length of the reminding list (default: 3)

    remind_mode: string
        The postion of the remining items in recommendation list (top, end). (default: end)


    '''

    def __init__(self, steps=10, weighting='div', pruning=20, last_n_days=None, idf_weight=False, last_in_session=False, session_weighting='div',
                 boost_own_sessions=None,
                 reminders=False, remind_strategy='recency', remind_sessions_num=6, reminders_num=3, remind_mode='end', weight_base=1, weight_IRec=0,
                 session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId'):
        self.steps = steps
        self.pruning = pruning
        self.weighting = weighting
        self.session_weighting = session_weighting
        self.last_n_days = last_n_days
        self.idf_weight = idf_weight
        self.last_in_session = last_in_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.session = -1
        self.session_items = []
        # user_based
        self.user_key = user_key
        self.boost_own_sessions = boost_own_sessions

        self.hasReminders = reminders
        if self.hasReminders:
            if remind_strategy == 'hybrid':
                self.reminder = Reminder(remind_strategy=remind_strategy, remind_sessions_num=remind_sessions_num,
                                         weight_base=weight_base, weight_IRec=weight_IRec)
            else:  # basic reminders
                self.reminder = Reminder(remind_strategy=remind_strategy, remind_sessions_num=remind_sessions_num,
                                         reminders_num=reminders_num, remind_mode=remind_mode)

    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).


        '''

        if self.last_n_days != None:

            max_time = dt.fromtimestamp(data[self.time_key].max())
            date_threshold = max_time.date() - td(self.last_n_days)
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[data[self.time_key] >= stamp]

        else:
            train = data

        if self.idf_weight:
            self.idf = self.compute_idf(data, item_key=self.item_key, session_key=self.session_key)

        cur_session = -1
        last_items = []
        rules = dict()
        # In SR: rule-set is a dic like: {item_a: {item_b: score}, item_b: {item_c: score, item_d: score, item_a: score}}
        # In user-based SR: rule-set is a dic like: {item_a: {item_b: [score, {userId1}]}, item_b: {item_c: [score, {userId_1, userId_2}], item_d: [score, {userId_2, userId_3}], item_a: [score, {userId_4}]}}
            # fist element of the list is : score, the rest is a SET of user ids who had this rule in their past sessions

        # get the position of the columns
        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_user = train.columns.get_loc(self.user_key)  # user_based

        for row in train.itertuples(index=False):

            session_id, item_id, user_id = row[index_session], row[index_item], row[index_user]

            if session_id != cur_session:
                cur_session = session_id
                last_items = []
            else:
                for i in range(1, self.steps + 1 if len(last_items) >= self.steps else len(last_items) + 1):
                    prev_item = last_items[-i]

                    if not prev_item in rules:
                        rules[prev_item] = dict()


                    if not item_id in rules[prev_item]:
                        userSet = set()
                        rules[prev_item][item_id] = [0, userSet]

                    if not user_id in rules[prev_item][item_id][1]: # in userSet
                        rules[prev_item][item_id][1].add(user_id)

                    weight = getattr(self, self.weighting)(i)
                    if self.idf_weight:
                        if self.idf_weight == 1:
                            weight *= self.idf[prev_item]
                        elif self.idf_weight == 2:
                            weight += self.idf[prev_item]

                    rules[prev_item][item_id][0] += weight

            last_items.append(item_id)

            # reminders
            if self.hasReminders:  # user_based  # for 'session_similarity' or 'recency'
                self.reminder.reminders_fit_in_loop(row, index_user, index_session, index_item)
                # prev_s_id = self.reminder.reminders_fit_in_loop(row, index_user, index_session, index_item, prev_s_id)

        if self.pruning > 0:
            self.prune(rules)

        self.rules = rules

        # reminders
        if self.hasReminders:  # user_based
            self.reminder.reminders_fit(train, self.user_key, self.item_key, self.time_key)

    #         print( 'Size of map: ', asizeof.asizeof(self.rules))

    def linear(self, i):
        return 1 - (0.1 * i) if i <= 100 else 0

    def same(self, i):
        return 1

    def div(self, i):
        return 1 / i

    def log(self, i):
        return 1 / (log10(i + 1.7))

    def quadratic(self, i):
        return 1 / (i * i)

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids, skip=False, mode_type='view', timestamp=0):

        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        if session_id != self.session:
            self.session_items = []
            self.session = session_id

        if mode_type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        preds = np.zeros(len(predict_for_item_ids))

        # useless: add extend_session_length to make predictions
        if input_item_id in self.rules:
            for key in self.rules[input_item_id]:
                # preds[predict_for_item_ids == key] = self.rules[input_item_id][key]
                preds[predict_for_item_ids == key] = self.rules[input_item_id][key][0]
                if self.boost_own_sessions is not None and self.boost_own_sessions > 0.0 and input_user_id in self.rules[input_item_id][key][1]: # if the rule also belong to the same user_id, then boost its score!
                    preds[predict_for_item_ids == key] = preds[predict_for_item_ids == key] + self.rules[input_item_id][key][0] * self.boost_own_sessions


        if self.last_in_session:
            for i in range(2, self.last_in_session + 2):
                if len(self.session_items) >= i:
                    item = self.session_items[-i]
                    if item in self.rules:
                        for key in self.rules[item]:
                            preds[predict_for_item_ids == key] += self.rules[item][key] * getattr(self,
                                                                                                  self.session_weighting)(i)
                else:
                    break

        # test
        #         for i in range(2,4):
        #             if len(self.session_items) >= i :
        #                 item = self.session_items[-i]
        #                 for key in self.rules[ item ]:
        #                     preds[ predict_for_item_ids == key ] += self.rules[item][key] * (1/i)

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        if self.hasReminders:  # user_based
            if self.reminder.remind_strategy == 'hybrid':
                series = self.reminder.reminders_predict_next(input_user_id, series, self.item_key,
                                                              self.time_key, input_timestamp=timestamp)
            else:  # basic reminders
                series = self.reminder.reminders_predict_next(input_user_id, series, self.item_key, self.time_key)

        return series

    def prune(self, rules):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int(len(tmp) * self.pruning)
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter(tmp)
            rules[k1] = dict()
            for k2, v in counter.most_common(keep):
                rules[k1][k2] = v

    def compute_idf(self, train, item_key="ItemId", session_key="SessionId"):

        idf = pd.DataFrame()
        idf['idf'] = train.groupby(item_key).size()
        idf['idf'] = np.log(train[session_key].nunique() / idf['idf'])
        idf['idf'] = (idf['idf'] - idf['idf'].min()) / (idf['idf'].max() - idf['idf'].min())
        idf = idf['idf'].to_dict()

        return idf

    def clear(self):
        self.rules = {}

    def support_users(self):
        '''
            whether it is a session-based or session-aware algorithm
            (if returns True, method "predict_with_training_data" must be defined as well)

            Parameters
            --------

            Returns
            --------
            True : if it is session-aware
            False : if it is session-based
        '''
        return True

    def predict_with_training_data(self):
        '''
            (this method must be defined if "support_users is True")
            whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

            Parameters
            --------

            Returns
            --------
            True : e.g. hgru4rec
            False : e.g. uvsknn
            '''
        return False