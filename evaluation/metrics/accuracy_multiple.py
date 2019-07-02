import numpy as np
    
class Precision: 
    '''
    Precision( length=20 )

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
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += self.length
        self.hit += len( set(next_item) & set(result[:self.length].index) )
    
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / self.length
        
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
        return ("Precision@" + str(self.length) + ": "), (self.hit/self.test)
    
class Recall: 
    '''
    Precision( length=20 )

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
        a=set(next_item)
        b=set(result[:self.length].index)
        c=set(next_item) & set(result[:self.length].index)
        self.hit += len( set(next_item) & set(result[:self.length].index) )
       
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / len(next_items)
        
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
        return ("Recall@" + str(self.length) + ": "), (self.hit/self.test)

class MAP: 
    '''
    MAP( length=20 )

    Used to iteratively calculate the mean average precision for a result list with the defined length. 

    Parameters
    -----------
    length : int
        MAP@length
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
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add_multiple(self, result, next_items, for_item=0, session=0,position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        last_recall = 0
        
        res = 0
        
        for i in range(self.length):
            recall = self.recall(result[:i].index, next_items)
            precision = self.precision(result[:i].index, next_items)
            res += precision * (recall - last_recall)
            last_recall = recall
        
        self.pos += res
        self.test += 1
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        sum = 0
        
        for i in range(self.length):
            sum += self.mrr(result, next_item, i+1)
        
        self.pos += ( sum / self.length )
        self.test += 1
    
    def recall( self, result, next_items ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        return len( set(next_items) & set(result) ) / len( next_items )
    
    def precision( self, result, next_items ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        return len( set(next_items) & set(result) ) / self.length
    
    def mrr( self, result, next_item, n ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        res = result[:n]
        
        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            return 1.0/rank
        else:
            return 0
        
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
        return ("MAP@" + str(self.length) + ": "), (self.pos/self.test)

class NDCG:
    '''
    NDCG( length=20 )

    Used to iteratively calculate the Normalized Discounted Cumulative Gain for a result list with the defined length.

    Parameters
    -----------
    length : int
        NDCG@length
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
        self.test = 0;
        self.pos = 0

    def skip(self, for_item=0, session=-1):
        pass


    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        dcg = self.dcg(result[:self.length].index, next_items)
        dcg_max = self.dcg(next_items[:self.length], next_items)

        self.pos += dcg/dcg_max
        self.test += 1


    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.add_multiple(result, [next_item])


    def dcg(self, result, next_items):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''

        # relatedItems = list(set(result) & set(next_items))
        # for i in range(len(relatedItems)):
        #     idx = list(result).index(relatedItems[i])+1 #ranked position = index+1
        #     if idx == 1:
        #         res += rel
        #     else:
        #         res += rel / np.log2(idx)

        res = 0;
        rel = 1;
        ranked_list_len = min(len(result), self.length)

        next_items = set(next_items)
        for i in range(ranked_list_len):          #range(self.length):
            if result[i] in next_items:
                if i == 0:
                    res += rel
                else:
                    res += rel / np.log2(i+1)

        # res = rel[0]+np.sum(rel[1:] / np.log2(np.arange(2, rel.size + 1)))
        return res


    def sortFunc(e):
        return e.values;



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
        i = 0
        for part, series in result.iteritems():
            result.sort_values(part, ascending=False, inplace=True)
            self.add(series, next_item[i])
            i += 1

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("NDCG@" + str(self.length) + ": "), (self.pos/self.test)


class NDCG_relevance:
    '''
    NDCG( length=20 )

    Used to iteratively calculate the Normalized Discounted Cumulative Gain for a result list with the defined length.

    Parameters
    -----------
    length : int
        NDCG@length
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
        self.train=train
        return

    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test = 0;
        self.pos = 0

    def skip(self, for_item=0, session=-1):
        pass

    def set_buys(self, buys,test_set):
        self.buys=buys
        self.test_set=test_set
        buys_filterd = buys[buys['SessionId'].isin(self.train['SessionId'])]
        self.ratio_buys=len(buys_filterd)/len(self.train)
        return

    def add_multiple(self, result, next_items, for_item=0, session=0,position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''

        dcg = self.dcg(result[:self.length].index, next_items,session,position)
        dcg_max = self.dcg(next_items[:self.length], next_items,session,position)

        self.pos += dcg/dcg_max
        self.test += 1


    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        self.add_multiple(result, [next_item],session)


    def dcg(self, result, next_items,session,position):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.

        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''

        # relatedItems = list(set(result) & set(next_items))
        # for i in range(len(relatedItems)):
        #     idx = list(result).index(relatedItems[i])+1 #ranked position = index+1
        #     if idx == 1:
        #         res += rel
        #     else:
        #         res += rel / np.log2(idx)

        res = 0;
        rel = 1;
        rel_buy=self.ratio_buys
        rel_count_next_items=1
        rel_click=1
        ranked_list_len = min(len(result), self.length)

        next_items=list(next_items)
        #next_items = set(next_items)
        for i in range(ranked_list_len):          #range(self.length):
            if result[i] in next_items:
                #if item is bought
                b=self.buys.loc[self.buys['SessionId'] == session].ItemId.values
                r=result[i]
                if result[i] in self.buys.loc[self.buys['SessionId'] == session].ItemId.values:
                    rel+=rel_buy
                    #print("in buys")

                #how many time item appers in the next_items list

                rel+=next_items.count(result[i])*rel_count_next_items


                #if item has been clicked before in the session
                session_rows=self.test_set.loc[self.test_set['SessionId'] == session]
                previous_items=session_rows.iloc[:position]
                if result[i] in  previous_items.ItemId.values:
                    #print("previous items")
                    rel+=rel_click

                if i == 0:
                    res += rel
                else:
                    res += rel / np.log2(i+1)

                rel=0

        # res = rel[0]+np.sum(rel[1:] / np.log2(np.arange(2, rel.size + 1)))
        return res


    def sortFunc(e):
        return e.values;



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
        i = 0
        for part, series in result.iteritems():
            result.sort_values(part, ascending=False, inplace=True)
            self.add(series, next_item[i])
            i += 1

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("NDCG_relevance@" + str(self.length) + ": "), (self.pos/self.test)