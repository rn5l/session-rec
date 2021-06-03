from .context_tree_BVMM import TreeRoot, History
from .context_tree_BVMM import StdExpert, DirichletExpert, BayesianMixtureExpert
from collections import deque, OrderedDict
import sys
import csv
import pickle
import time
import scipy.io
import random
import collections

import numpy as np
import pandas as pd
import time

from sklearn import preprocessing

class ContextTree:
    '''
    Code based on work by Mi et al., Context Tree for Adaptive Session-based Recommendation, 2018.

    Parameters
    --------
    history_maxlen: max considered context length

    nb_candidates (only used for adaptive configuration): the number of recent candidates considered for adaptive configuration

    expert: type of expert for each context
    '''
    def __init__(self, history_maxlen = 50, nb_candidates = 1000, expert = 'StdExpert', session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time'):
        self.history_maxlen = history_maxlen
        self.nb_candidates = nb_candidates
        self.expert = eval( expert )
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

        self.root = TreeRoot(self.expert)

        self.histories = History(history_maxlen)

        self.recent_candidates = OrderedDict()

        self.user_to_previous_recoms = {}

    def fit(self, train, items=None):
        '''
        fit training data for static evalution

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        start_time = time.time()
        for index, row in train.iterrows():
            self.fit_one_row(row, True)
            if index % 1000000 ==0:
                print(index, '---- Time: ', time.time()-start_time)

    def fit_one_row(self, row, first_in_session):

        '''
        fit one row for static setting
        '''

        current_session = row[self.session_key]
        current_item = row[self.item_key]

        history = self.histories.get_history(current_session)

        self.root.update(current_item, history)

        history.appendleft(current_item)

        self.root.expand(history)


    def fit_time_order_online(self, row, first_in_session):

        '''
        fit one row for adpative configuration, we data is ordered by time
        nb_candidates is used to keep a pool of recent candidates
        '''

        current_session = row[self.session_key]
        current_item = row[self.item_key]

        history = self.histories.get_history(current_session)

        self.root.update(current_item, history)

        history.appendleft(current_item)

        best_item_and_probas = self.root.get_n_most_probable(self.recent_candidates.keys(), history)
        predictions = [proba for rec, proba in best_item_and_probas]
        series = pd.Series(data=predictions, index=[int(rec) for rec, proba in best_item_and_probas])

        if not first_in_session:
            predictions = preprocessing.normalize([predictions])
            series = pd.Series(data=predictions[0], index=[int(rec) for rec, proba in best_item_and_probas])

        self.user_to_previous_recoms[current_session] = series

        self.root.expand(history)


        self.recent_candidates.pop(current_item, None)
        self.recent_candidates[current_item] = True
        if len(self.recent_candidates) > self.nb_candidates:
            self.recent_candidates.popitem(last=False)


    def match_context(self, row, items_to_predict, normalize):

        '''
        only used in static evaluation
        update the recommendation given next time with current row as input
        the model (CT) is not updated in static evalution for a fair comparison against other methods
        '''

        current_session = row[self.session_key]
        current_item = row[self.item_key]

        history = self.histories.get_history(current_session)

        history.appendleft(current_item)

        best_item_and_probas = self.root.get_n_most_probable(items_to_predict, history)
        predictions = [proba for rec, proba in best_item_and_probas]
        if normalize:
            predictions = preprocessing.normalize([predictions])
            #predictions = [i / sum(predictions) for i in predictions]
            series = pd.Series(data=predictions[0], index=[int(rec) for rec, proba in best_item_and_probas])
        else:
            series = pd.Series(data=predictions, index=[int(rec) for rec, proba in best_item_and_probas])
        self.user_to_previous_recoms[current_session] = series



    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0, skip=False, mode_type="view"):
        # print(input_item_id)
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        
        row = { self.item_key: input_item_id, self.session_key: session_id }
        self.match_context(row, predict_for_item_ids, False)

        previous_recoms = self.user_to_previous_recoms.get(session_id)
        #print(previous_recoms[:5])

        return previous_recoms

    def clear(self):
        
        del self.histories
        del self.recent_candidates
        del self.user_to_previous_recoms

    def support_users(self):
        '''
          whether it is a session-based or session-aware algorithm
          (if return True, method "predict_with_training_data" must be defined as well)

          Parameters
          --------

          Returns
          --------
          True : if it is session-aware
          False : if it is session-based
        '''
        return False
