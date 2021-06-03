from _operator import itemgetter
from math import sqrt
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
from cmath import log10

class ContextKNN:
    '''
    ContextKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

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

    def __init__( self, k, sample_size=20000, size=32, embeddings_file='emb/item_emb', sampling='recent', similarity='sim1', weighting='div', extend=False, normalize=True, weighting_score='div_score', weighting_time=False, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.k = k
        self.size = size
        self.embeddings_file = embeddings_file
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.weighting = weighting
        self.weighting_time = weighting_time
        self.weighting_score = weighting_score
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        
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
        
        filen = self.embeddings_file+'_'+str(self.size)+'.csv'
        if not os.path.exists( filen ):
            #create
            self.create_embeddings(train, self.size, filen)
        
        emb = pd.read_csv( filen )
        emb.index = emb[self.item_key]
        del emb[self.item_key]
        
        for item in emb.index:
            self.item_emb_map[item] = emb.ix[item].values
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        
        
        session = -1
        session_items = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session : time})
                    
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
            
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        
        
    def create_embeddings(self, train, size, filen):
        
        start = time.time()
        
        train[ self.item_key+'_str' ] = train[ self.item_key ].astype('str')
        sequences = train.groupby( self.session_key )[self.item_key+'_str'].apply(list)
        
        print('created data frame in ',(time.time() - start))
        
        start = time.time()
       
        model = gensim.models.Word2Vec(sequences, size=size, window=5, min_count=1, workers=4, iter=25, sg=1)
          
        d = {}  
        for word in model.wv.vocab:
            d[word] = model.wv[word]
        
        frame = pd.DataFrame( d )
        frame = frame.T
        frame.columns = ['lf_'+str(i) for i in range(size)]
        frame[ self.item_key ] = frame.index.astype( np.int32 )
        
        frame.to_csv( filen , index=False )
        
        print('created item2vec features in ',(time.time() - start))
    
      
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
        
        profile = self.create_profile( self.session_items )
        neighbors = self.find_neighbors( self.session_items, input_item_id, session_id, profile, [], timestamp )
        scores = self.score_items( neighbors, self.session_items, timestamp )
         
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
    
    
    def score_items(self, neighbors, current_session, timestamp):
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
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            step = 1
            
            for item in reversed( current_session ):
                if item in items:
                    decay = getattr(self, self.weighting_score)( step )
                    break
                step += 1
                                    
            for item in items:
                old_score = scores.get( item )
                similarity = session[1]
                
                if old_score is None:
                    scores.update({item : ( similarity * decay ) })
                else: 
                    new_score = old_score + ( similarity * decay )
                    scores.update({item : new_score})
                    
        return scores
    
       
    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def create_profile_simple( self, session_items):
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
               
        res = np.zeros(self.size)
        sum = 0
        
        for item in session_items:
            vec = self.item_emb_map[item]
            res += vec
            sum += 1
                    
        return res / sum
    
    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def create_profile( self, session_items):
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
        pos_map = {}
        length = len( session_items )
        
        count = 1
        for item in session_items:
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
        
        for item in session_items:
            vec = self.item_emb_map[item]
            res += pos_map[item] * vec
            sum += pos_map[item]
                    
        return res / sum
            
    def sim1(self, profile, session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        
        profile_s = self.create_profile_simple( session )          
        score = dm.cosine(profile, profile_s)
        score = (1 - score)     
                       
        return score
    
    def sim2(self, profile, session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        
        dist = 0
        
        for i in session:
            vec = self.item_emb_map[i]
            dist += (1 - dm.cosine(profile, vec))
                       
        return dist / len(session)
    
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
    
    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same_score(self, i):
        return 1
    
    def div_score(self, i):
        return 1/i
    
    def log_score(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic_score(self, i):
        return 1/(i*i)
    
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
                        

    def calc_similarity(self, session_items, sessions, profile, dwelling_times, timestamp ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            n_items = self.items_for_session( session )
            sts = self.session_time[session]
                        
            similarity = getattr(self, self.similarity)( profile, n_items )
            
            if similarity > 0:
                
                if self.weighting_time:
                    diff = timestamp - sts
                    days = round( diff/ 60 / 60 / 24 )
                    decay = pow( 7/8, days )
                    similarity *= decay
                
                #print("days:",days," => ",decay)
                
                neighbors.append((session, similarity))
                
        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id, profile, dwelling_times, timestamp ):
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
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors, profile, dwelling_times, timestamp )
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
        
    
    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    def vec_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_vec_map.get(session);
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()
      