from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd


class StrategicHybridSeq:
    '''
    StrategicHybrid(algorithms, weights)

    Use different algorithms depending on the length of the current session.

    Parameters
    --------
    algorithms : list
        List of algorithms to combine with a switching strategy.
    thresholds : float
        Proper list of session length thresholds.
        For [5,10] the first algorithm is applied until the session exceeds a length of 5 actions, the second up to a length of 10, and the third for the rest.
    fit: bool
        Should the fit call be passed through to the algorithms or are they already trained?

    '''

    def __init__(self, algorithms, thresholds, n_bins=5, fit=True, clearFlag=True, session_key='SessionId', item_key='ItemId'):

        self.algorithms = algorithms
        self.thresholds = thresholds

        self.n_bins = n_bins
        self.session_key = session_key
        self.item_key = item_key
        self.run_fit = fit
        self.clearFlag = clearFlag
    
    def init(self, train, test=None, slice=None):
        for a in self.algorithms: 
            if hasattr(a, 'init'):
                a.init( train, test, slice )
    
    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        if self.run_fit:
            for a in self.algorithms:
                a.fit(data)

        self.session = -1
        self.session_items = []
        self.session_seq = 0

        self.rules, self.max = self.markov(data, item_key=self.item_key, session_key=self.session_key)

        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc(self.item_key)

        seq_sum = 0
        seq_steps = 0
        prev_item = -1
        cur_session = -1

        all_seq = []

        for row in data.itertuples(index=False):

            session_id, item_id = row[index_session], row[index_item]

            if session_id != cur_session:
                cur_session = session_id
                seq_sum = 0
                seq_steps = 0
            else:
                if prev_item in self.rules and item_id in self.rules[prev_item]:
                    seq_sum += self.rules[prev_item][item_id] / self.max
                    seq_steps += 1

                    all_seq.append( seq_sum /seq_steps )

            prev_item = item_id

        seqseries = pd.Series( all_seq )
        self.bin_series ,self.bins = pd.qcut(seqseries, self.n_bins, duplicates='drop', retbins=True)

        self.algorithm_map = {}
        self.algorithm_map[-1] = self.algorithms[0]

        for i in range(len(self.thresholds)):
            thres_list = self.thresholds[i]
            for bin in thres_list:
                self.algorithm_map[bin] = self.algorithms[i]

        self.bincount = []

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False,timestamp=0):
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

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()
            self.session_seq = 0

        self.session_items.append(input_item_id)

        bin = -1
        if len( self.session_items ) > 1:
            prev_item = self.session_items[-2]
            item_id = self.session_items[-1]
            if prev_item in self.rules and item_id in self.rules[prev_item]:
                self.session_seq += self.rules[prev_item][item_id] / self.max

            current_seq = self.session_seq / (len(self.session_items)-1)
            bin = pd.cut( [current_seq], bins=self.bins ).codes[0]
            if current_seq > self.bin_series.cat.categories.max().right:
                bin = self.bin_series.cat.codes.max()
                self.bincount.append(-2)

        preds = self.algorithm_map[bin].predict_next(session_id, input_item_id, predict_for_item_ids, skip)

        self.bincount.append( bin )

        return preds


    def markov(self,data, session_key="SessionId", item_key="ItemId", time_key="Time", last_n_days=None):
        '''
            Trains the predictor.

            Parameters
            --------
            data: pandas.DataFrame
                Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
                It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).


            '''

        if last_n_days != None:

            max_time = dt.fromtimestamp(data[time_key].max())
            date_threshold = max_time.date() - td(last_n_days)
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[data[time_key] >= stamp]

        else:
            train = data

        cur_session = -1
        prev_item = -1
        rules = dict()

        index_session = train.columns.get_loc(session_key)
        index_item = train.columns.get_loc(item_key)

        max = -1

        for row in train.itertuples(index=False):

            session_id, item_id = row[index_session], row[index_item]

            if session_id != cur_session:
                cur_session = session_id
            else:
                if not prev_item in rules:
                    rules[prev_item] = dict()

                if not item_id in rules[prev_item]:
                    rules[prev_item][item_id] = 0

                rules[prev_item][item_id] += 1

                if rules[prev_item][item_id] > max:
                    max = rules[prev_item][item_id]

            prev_item = item_id

        # if self.pruning > 0:
        #     self.prune(rules)

        return rules, max


    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()
