import numpy as np
import pandas as pd
from math import log10
import collections as col

class AssociationRules: 
    '''
    SequentialRules(steps = 3, weighting='div', pruning=0.0)
        
    Parameters
    --------
    steps : int
        TODO. (Default value: 3)
    weighting : string
        TODO. (Default value: 3)
    pruning : float
        TODO. (Default value: 20)
    
    '''
    
    def __init__( self, pruning=20, session_key='SessionId', item_key='ItemId' ):
        self.pruning = pruning
        self.session_key = session_key
        self.item_key = item_key
        self.session = -1
        self.session_items = []
            
    def fit( self, data, test=None ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        
            
        '''

        cur_session = -1
        last_items = []
        rules = dict()
        
        index_session = data.columns.get_loc( self.session_key )
        index_item = data.columns.get_loc( self.item_key )
        
        for row in data.itertuples( index=False ):
            
            session_id, item_id = row[index_session], row[index_item]
            
            if session_id != cur_session:
                cur_session = session_id
                last_items = []
            else: 
                for item_id2 in last_items: 
                    
                    if not item_id in rules :
                        rules[item_id] = dict()
                    
                    if not item_id2 in rules :
                        rules[item_id2] = dict()
                    
                    if not item_id in rules[item_id2]:
                        rules[item_id2][item_id] = 0
                    
                    if not item_id2 in rules[item_id]:
                        rules[item_id][item_id2] = 0
                    
                    rules[item_id][item_id2] += 1
                    rules[item_id2][item_id] += 1
                    
            last_items.append( item_id )
        
        if self.pruning > 0 :
            self.prune( rules )
            
        self.rules = rules
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 10 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
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
        
        if type == 'view':
            self.session_items.append( input_item_id )
            
        if skip:
            return
        
        preds = np.zeros( len(predict_for_item_ids) ) 
             
        if input_item_id in self.rules:
            for key in self.rules[input_item_id]:
                preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]
        
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
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v              
                
    def clear(self):
        self.rules = {}