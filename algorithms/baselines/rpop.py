import numpy as np
import pandas as pd
from datetime import datetime as dt
import datetime

class RPop:
    '''
    RPop(top_n=100, item_key='ItemId', support_by_key=None)
    
    Popularity predictor that gives higher scores to items with larger support.
    
    The score is given by:
    
    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    num_days : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    time_key : string
        The header of the timestamp column in the training data. (Default value: 'Time')
    
    '''
    
    def __init__(self, num_days = 1, item_key = 'ItemId', time_key = 'Time'):
        self.num_days = num_days
        self.item_key = item_key
        self.time_key = time_key
            
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        max_time = dt.fromtimestamp( data[self.time_key].max() )
        date_threshold = max_time.date() - datetime.timedelta( self.num_days )
        stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
        self.pop_list = data[ data[self.time_key] >= stamp ].groupby( self.item_key ).size()
        self.max_pop = self.pop_list.max();
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
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
        if not input_item_id in self.pop_list.index:
            self.pop_list.set_value( input_item_id, 0 )
        
        self.pop_list[input_item_id] += 1
        if self.pop_list[input_item_id] > self.max_pop :
            self.max_pop = self.pop_list[input_item_id]
        
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]] / self.max_pop
        return pd.Series(data=preds, index=predict_for_item_ids)
        