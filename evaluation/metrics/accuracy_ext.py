import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td

class AccuracyBase:

    def set_rules(self, rules, max):
        self.rules = rules
        self.max = max

    def set_popularity(self, popscore):
        self.popscore = popscore

    def set_idf(self, idf):
        self.idf = idf

    def set_buys(self, buys,test_set):
        self.buys=buys
        self.test_set=test_set
        return

    def add_to_table(self,res,session,position):
        session_items = self.test_set.loc[self.test_set['SessionId'] == session].ItemId.values[:position + 1]
        seq = []
        for i in range(len(session_items) - 1):
            if session_items[i] in self.rules and session_items[i + 1] in self.rules[session_items[i]]:
                seq.append(self.rules[session_items[i]][session_items[i + 1]])

        mean_seq = np.array(seq).mean()

        seq_pop=[]
        for i in range(len(session_items)):
            seq_pop.append(self.popscore[session_items[i]])

        mean_seq_pop = np.array(seq_pop).mean()

        seq_idf=[]
        for i in range(len(session_items)):
            seq_idf.append(self.idf[session_items[i]])

        mean_seq_idf = np.array(seq_idf).mean()


        self.table['Value'].append( res )
        self.table['SessionSeq'].append(mean_seq)
        self.table['SessionPop'].append(mean_seq_pop)
        self.table['SessionIdf'].append(mean_seq_idf)


class MRR(AccuracyBase):
    '''
    MRR( length=20 )

    Used to iteratively calculate the average mean reciprocal rank for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MRR@length
    '''
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0

        self.table = {}
        self.table['SessionPop'] = []
        self.table['SessionSeq'] = []
        self.table['SessionIdf'] = []
        self.table['Value'] = []
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        preds = result[:self.length]
        
        self.test += 1

        res = 0
        if next_item in preds.index:
            rank = preds.index.get_loc( next_item )+1
            res = ( 1.0/rank )
            self.pos += res

        self.add_to_table(res,session,position)
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test), pd.DataFrame( self.table )
    
class HitRate(AccuracyBase):
    '''
    MRR( length=20 )

    Used to iteratively calculate the average hit rate for a result list with the defined length. 

    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''

        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0

        self.table = {}
        self.table['SessionPop'] = []
        self.table['SessionSeq'] = []
        self.table['SessionIdf'] = []
        self.table['Value'] = []

    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''


        self.test += 1
        res = 0
        if next_item in result[:self.length].index:
            res = 1
            self.hit += res

        self.add_to_table(res,session,position)

        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test), pd.DataFrame( self.table )