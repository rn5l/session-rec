from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np


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

        self.idf = self.compute_idf(data, item_key=self.item_key, session_key=self.session_key)

        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc(self.item_key)

        idf_sum = 0
        idf_steps = 0
        prev_item = -1
        cur_session = -1

        all_idf = []

        for row in data.itertuples(index=False):

            session_id, item_id = row[index_session], row[index_item]

            if session_id != cur_session:
                cur_session = session_id
                idf_sum = 0
                idf_steps = 0
            
            idf_sum += self.idf[item_id] if item_id in self.idf else 0
            idf_steps += 1
            
            all_idf.append( idf_sum /idf_steps )
            
            prev_item = item_id

        idfseries = pd.Series( all_idf )
        self.bin_series ,self.bins = pd.qcut(idfseries, self.n_bins, duplicates='drop', retbins=True)

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
            self.session_idf = 0

        self.session_items.append(input_item_id)

        bin = -1
        
        item_idf = self.idf[ input_item_id ] if input_item_id in self.idf else 0
        self.session_idf += item_idf
        current_idf = self.session_idf / len(self.session_items)

        bin = pd.cut([current_idf], bins=self.bins).codes[0]
        if current_idf > self.bin_series.cat.categories.max().right:
            bin = self.bin_series.cat.codes.max()
            self.bincount.append(-2)

        preds = self.algorithm_map[bin].predict_next(session_id, input_item_id, predict_for_item_ids, skip)

        self.bincount.append(bin)

        return preds
        
    def compute_idf( self, train, item_key="ItemId", session_key="SessionId" ):
        
        idf = pd.DataFrame()
        idf['idf'] = train.groupby(item_key).size()
        idf['idf'] = np.log(train[session_key].nunique() / idf['idf'])
        idf = idf['idf'].to_dict()
    
        return idf
        
        
    def clear(self):
        if(self.clearFlag):
            for a in self.algorithms:
                a.clear()
