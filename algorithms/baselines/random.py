import numpy as np
import pandas as pd

class RandomPred:
    '''
    RandomPred()
    
    Initializes a random predcitor, which is a baseline predictor that gives back a random score for each item.  
    
    '''
    def fit(self, data):
        '''
        Dummy function for training.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        pass

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        return pd.Series(data=np.random.rand(len(predict_for_item_ids)), index=predict_for_item_ids)
    