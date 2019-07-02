class ArtistCoherence:
    '''
    Coverage( length=20 )

    Used to iteratively calculate the coverage of an algorithm regarding the item space. 

    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    item_key = 'ItemId'
    artist_key = 'ArtistId'
    
    def __init__(self, length=20):
        self.length = length
        self.average = 0.0
        self.count = 0.0
        
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        index_item = train.columns.get_loc( self.item_key )
        index_artist = train.columns.get_loc( self.artist_key )
        
        self.item_artist = {}
        for row in train.itertuples(index=False):
            self.item_artist[ row[index_item] ] = row[index_artist]
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.session = set() #artist set
        self.session_id = -1
        self.average = 0.0
        self.count = 0.0
    
    def skip(self, for_item = 0, session = -1 ):
        pass
    
    def add(self, result, next_item, for_item=0, session=-1, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        recs = result[:self.length]
        
        if self.session_id != session:
            self.session = set()
        
        self.session.add( self.item_artist[for_item] )
        
        rec_artists = set()
        for item in recs.index: 
            rec_artists.add( self.item_artist[ item ] )
            
        union = len( self.session | rec_artists )
        intersect = len( self.session & rec_artists )
        
        self.average += ( intersect / union ) if union > 0 else 0
        self.count += 1


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
        return ("Coherence@" + str(self.length) + ": "), ( self.average / self.count )
    