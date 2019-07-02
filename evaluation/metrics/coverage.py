
class Coverage:
    '''
    Coverage( length=20 )

    Used to iteratively calculate the coverage of an algorithm regarding the item space. 

    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    item_key = 'ItemId'
    
    def __init__(self, length=20):
        self.num_items = 0
        self.length = length
        self.time = 0;
        
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''        
        self.coverage_set = set()
        self.num_items = len( train[self.item_key].unique() )
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.coverage_set = set()
        return
    
    def skip(self, for_item = 0, session = -1 ):
        pass
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        recs = result[:self.length]
        items = recs.index.unique()
        self.coverage_set.update( items )
        
    def add_multiple(self, result, next_items, for_item=0, session=0, position=None):   
        self.add(result, next_items[0], for_item, session)
        
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
        return ("Coverage@" + str(self.length) + ": "), ( len(self.coverage_set) / self.num_items )
    