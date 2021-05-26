from _operator import itemgetter
from math import sqrt, log10
import random
import time
import numpy as np
import pandas as pd
import os
import psutil
import gc
import gensim
from sympy.external.tests.test_scipy import scipy
import scipy.spatial.distance as dm
from scipy import sparse

class ItemKNN:
    '''
    ItemKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')
    
    
    
    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=20000, size=32, embeddings_file='emb/item_emb', sampling='recent', similarity='cosine', weighting='div', extend=False, normalize=True, last_n=3, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.k = k
        self.size = size
        self.embeddings_file = embeddings_file
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.weighting = weighting
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.last_n = last_n
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.item_emb_map = dict()
        self.session_time = dict()
        
        self.sim_time = 0
        
    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        filen = self.embeddings_file+'_word2vec_'+str(self.size)+'.csv'
        if not os.path.exists( filen ):
            #create
            self.create_embeddings(train, self.size, filen)
            
        filen = self.embeddings_file+'_als_'+str(self.size)+'.csv'
        if not os.path.exists( filen ):
            #create
            self.create_embeddings_als(train, self.size, filen)
        
        emb = pd.read_csv( filen )
        emb.index = emb[self.item_key]
        del emb[self.item_key]
        
        for item in emb.index:
            self.item_emb_map[item] = emb.ix[item].values
        
        newest = pd.DataFrame()
        newest[self.time_key] = train.groupby( [self.item_key] )[self.time_key].max()
        newest[self.item_key] = newest.index
        
        newest.sort_values( by=self.time_key, ascending=False, inplace=True )
        self.newest = newest.head(self.sample_size)
        
    def create_embeddings(self, train, size, filen):
        
        start = time.time()
        
        train[ self.item_key+'_str' ] = train[ self.item_key ].astype('str')
        sequences = train.groupby( self.session_key )[self.item_key+'_str'].apply(list)
        
        print('created data frame in ',(time.time() - start))
        
        start = time.time()
       
        model = gensim.models.Word2Vec(sequences, size=size, window=6, min_count=1, workers=4, iter=10, sg=0)
          
        d = {}  
        for word in model.wv.vocab:
            d[word] = model.wv[word]
        
        frame = pd.DataFrame( d )
        frame = frame.T
        frame.columns = ['lf_'+str(i) for i in range(size)]
        frame[ self.item_key ] = frame.index.astype( np.int32 )
        
        frame.to_csv( filen , index=False )
        
        print('created item2vec features in ',(time.time() - start))
    
    def create_embeddings_als(self, train, size, filen, user='SessionId', item='ItemId'):
        
        start = time.time()
        
        combi = train[ [ user, item ] ]
        combi.drop_duplicates( keep='first' )
        combi['value'] = 1.0
        
        umap = pd.Series( index=combi[user].unique(), data=range(combi[user].nunique()) )
        imap = pd.Series( index=combi[item].unique(), data=range(combi[item].nunique()) )
            
        tstart = time.time()
        
        combi = pd.merge(combi, pd.DataFrame({item:imap.index, 'iid':imap[imap.index].values}), on=item, how='inner')
        combi = pd.merge(combi, pd.DataFrame({user:umap.index, 'uid':umap[umap.index].values}), on=user, how='inner')
        
        print( 'add index in {}'.format( (time.time() - tstart) ) )
        
        SPM = sparse.csr_matrix(( combi['value'].tolist(), (combi.iid, combi.uid )), shape=( combi[item].nunique(), combi[user].nunique() ))
        
        print( 'matrix in {}'.format( (time.time() - tstart) ) )
        
        print( 'created user features in ',(time.time() - start) )
        
        start = time.time()
        
        model = implicit.als.AlternatingLeastSquares( factors=size, iterations=10 )
    
        # train the model on a sparse matrix of item/user/confidence weights
        model.fit(SPM)
        
        Ifv =  model.item_factors
        
        SF = ['lf_'+str(i) for i in range(size)]
        
        If = pd.DataFrame( Ifv, index=imap.index )
        If.columns = SF
        If[item] = If.index
         
        If.to_csv( filen, index=False )
        
        print('created als item  features in ',(time.time() - start))
        
        
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, timestamp=0, skip=False, type='view'):
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
        
#         gc.collect()
#         process = psutil.Process(os.getpid())
#         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')
        
        if( self.session != session_id ): #new session
            
            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()
        
        if type == 'view':
            self.session_items.append( input_item_id )
        
        if skip:
            return
        
        profile = self.create_profile( self.session_items, session_id, self.last_n )       
        scores = self.score_items( profile )
         
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        
        items = predict_for_item_ids[mask]
        values = [scores[x] if x in scores else 0 for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        
        if self.normalize:
            series = series / series.max()
        
        return series 
    
       
    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def create_profile( self, session_items, session_id, last_n=3):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        
        if last_n != None:
            items = session_items[-last_n:] if len(session_items) >= last_n else session_items
        else: 
            items = session_items
        
        pos_map = {}
        length = len( items )
        
        count = 1
        for item in items:
            if self.weighting is not None: 
                pos_map[item] = getattr(self, self.weighting)( count, length )
                count += 1
            else:
                pos_map[item] = 1
            
#         dt = dwelling_times.copy()
#         dt.append(0)
#         dt = pd.Series(dt, index=session_items)  
#         dt = dt / dt.max()
#         #dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
#         dt[session_items[-1]] = 1
#         
#         if self.dwelling_time:
#             #print(dt)
#             for i in range(len(dt)):
#                 pos_map[session_items[i]] *= dt.iloc[i]
#             #print(pos_map) 
               
        res = np.zeros(self.size)
        sum = 0
        
        for item in items:
            vec = self.item_emb_map[item]
            res += pos_map[item] * vec
            sum += pos_map[item]
#             res += vec
#             sum += 1

                    
        return res / sum
    
            
    def score_items(self, profile):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for item in self.newest.itertuples():
            
            vec = self.item_emb_map[item[0]]
            score = dm.cosine(vec, profile)
            
#             print('sim: ')
#             print( profile )
#             print( vec )
#             print( score )
#             print( (1 - score) )
            
            score = (1 - score)
            scores.update({item[0] : score})
                    
        return scores
    
    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0
    
    def same(self, i, length):
        return 1
    
    def div(self, i, length):
        return i/length
    
    def log(self, i, length):
        return 1/(log10((length-i)+1.7))
    
    def quadratic(self, i, length):
        return (i/length)**2   
    