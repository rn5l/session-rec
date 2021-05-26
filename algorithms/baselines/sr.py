import numpy as np
import pandas as pd
from math import log10
import collections as col
from datetime import datetime as dt
from datetime import timedelta as td

from pympler import asizeof
import sys


class SequentialRules:
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

    '''

    def __init__(self, steps=10, weighting='div', pruning=20, last_n_days=None, idf_weight=False, last_in_session=False,
                 session_weighting='div',
                 session_key='SessionId', item_key='ItemId', time_key='Time'):
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

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)

        for row in train.itertuples(index=False):

            session_id, item_id = row[index_session], row[index_item]

            if session_id != cur_session:
                cur_session = session_id
                last_items = []
            else:
                for i in range(1, self.steps + 1 if len(last_items) >= self.steps else len(last_items) + 1):
                    prev_item = last_items[-i]

                    if not prev_item in rules:
                        rules[prev_item] = dict()

                    if not item_id in rules[prev_item]:
                        rules[prev_item][item_id] = 0

                    weight = getattr(self, self.weighting)(i)
                    if self.idf_weight:
                        if self.idf_weight == 1:
                            weight *= self.idf[prev_item]
                        elif self.idf_weight == 2:
                            weight += self.idf[prev_item]
                    rules[prev_item][item_id] += weight

            last_items.append(item_id)

        if self.pruning > 0:
            self.prune(rules)

        self.rules = rules

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

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view',
                     timestamp=0):
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

        if input_item_id in self.rules:
            for key in self.rules[input_item_id]:
                preds[predict_for_item_ids == key] = self.rules[input_item_id][key]

        if self.last_in_session:
            for i in range(2, self.last_in_session + 2):
                if len(self.session_items) >= i:
                    item = self.session_items[-i]
                    if item in self.rules:
                        for key in self.rules[item]:
                            preds[predict_for_item_ids == key] += self.rules[item][key] * getattr(self,
                                                                                                  self.session_weighting)(
                                i)
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
        return False