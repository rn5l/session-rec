import theano.misc.pkl_utils as pickle
import pandas as pd
import numpy as np

class ResultFile:
    '''
    FileModel( modelfile )
    Uses a trained algorithm, which was pickled to a file.

    Parameters
    -----------
    modelfile : string
        Path of the model to load

    '''

    def __init__(self, file):
        # config.experimental.unpickle_gpu_on_cpu = True
        self.file = file
    
    def init(self, train, test=None, slice=None):
        file = self.file + ( ('.' + str(slice) + '.csv') if slice is not None else '' )
        if not '.csv' in file: 
            file = file + '.csv'
        self.recommendations = pd.read_csv( file, sep=';' )
                
        return
              
    def fit(self, train, test=None):
        
        self.pos = 0
        self.session_id = -1
        
        return

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
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
        
        if session_id != self.session_id:
            self.pos = 0
            self.session_id = session_id
        
        recs = self.recommendations[(self.recommendations.SessionId == session_id) & (self.recommendations.Position == self.pos) ]
        if len(recs) == 0: 
            recs = self.recommendations[self.recommendations.SessionId == session_id]
            recs = recs.iloc[[self.pos]]
        items = recs.Recommendations.values[0]
        scores = recs.Scores.values[0]
        
        def convert( data, funct ):
            return map( lambda x: funct(x), data.split(',') )
        
        items = convert( items, int )
        scores = convert( scores, float )
        
        res = pd.Series( index=items, data=scores ) 
        
        self.pos += 1
        
        return res
    
    def clear(self):
        del self.recommendations
