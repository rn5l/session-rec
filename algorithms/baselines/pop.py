import numpy as np
import pandas as pd

class Pop:
    '''
    Pop(top_n=100, item_key='ItemId', support_by_key=None)
    
    Popularity predictor that gives higher scores to items with larger support.
    
    The score is given by:
    
    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    support_by_key : string or None
        If not None, count the number of unique values of the attribute of the training data given by the specified header. If None, count the events. (Default value: None)
    
    '''
    
    def __init__(self, top_n = None, item_key = 'ItemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        if self.top_n is not None:
            self.pop_list.sort_values(ascending=False, inplace=True)
            self.pop_list = self.pop_list.head( self.top_n )  
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None):
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
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)
        