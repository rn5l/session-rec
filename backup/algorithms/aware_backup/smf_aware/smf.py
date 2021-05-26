from _operator import itemgetter
from math import sqrt
import random
import time

from pympler import asizeof
import numpy as np
import pandas as pd
from math import log10
import scipy.sparse
from scipy.sparse.csc import csc_matrix
import theano
import theano.tensor as T
from datetime import datetime as dt
from datetime import timedelta as td

class SessionMF:
    '''
    RecurrentNeigborhoodModel( learning_rate=0.01, regularization=0.001, session_key = 'SessionId', item_key= 'ItemId', time_key= 'ItemId')

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

    def __init__( self, factors=100, batch=50, learn='adam', learning_rate=0.001, regularization=0.0001, activation='linear', epochs=2, last_n_days=None, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.factors = factors
        self.batch = batch
        self.learning_rate = learning_rate
        self.learn = learn
        self.regularization = regularization
        self.epochs = epochs
        self.last_n_days = last_n_days
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        
        self.item_map = dict()
        self.item_count = 0
        self.session_map = dict()
        self.session_count = 0
        
        self.floatX = 'float32'
        self.intX = 'int32'
        
    def fit(self, data, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        self.unique_items = set(data[self.item_key].unique())
        self.num_items = data[self.item_key].nunique()
        self.item_list = np.zeros( self.num_items )
        
        start = time.time()
        self.init_items(data)
        print( 'finished init item map in {}'.format(  ( time.time() - start ) ) )
        
        if self.last_n_days != None:
            
            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]
        
        else: 
            train = data
            
        start = time.time()
        self.init_model(train)
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        session = -1
        session_items = np.zeros( self.num_items, dtype=np.float32 )
        session_count = 0
        
        batch_count = 0
        
        session_mat = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
        pos = np.zeros( self.batch ).astype( self.intX )
        neg = np.zeros( self.batch ).astype( self.intX )
        
        start = time.time()
                
        for i in range( self.epochs ):
            
            status = 0
            loss = 0
            count = 0
            hit = 0
            
            for row in train.itertuples(index=False):
                
                cs = row[index_session]
                ci = row[index_item]
                                 
                # cache items of sessions
                if cs != session:
            
                    #self.h_t.set_value( np.zeros( (self.num_items) ).astype( self.floatX ) , borrow=True)
                    session_items = np.zeros( self.num_items, dtype=np.float32 )
                    session_count = 1
                    
                else: 
                    #bi = random.choice(tuple(self.unique_items-set([ci])))
                    
                    #session_items -> ci > bi
                    pos[batch_count] = self.item_map[ci]
                    #neg[batch_count] = self.item_map[bi]
                                        
                    session_mat[batch_count] = session_items / (session_count-1)
                    
#                     preds = self.predict(session_items)
#                     val_pos = preds[self.item_map[ci]]
#                     
#                     if len( preds[preds > val_pos] ) < 20:
#                         hit += 1
                    
                    count += 1
                    batch_count += 1
                

                if batch_count == self.batch-1:
                    neg = np.random.randint(self.num_items, size=self.batch)
                    loss += self.train_model_batch( session_mat, pos, neg )
                    session_mat = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
                    pos = np.zeros( self.batch ).astype( self.intX )
                    batch_count = 0
                   
                session = cs
                session_items[ self.item_map[ci] ] = session_count
                
                status += 1
                session_count += 1
                
                if status % 10000 == 0 :
                    print( 'finished {} of {} in epoch {} with loss {} / hr {} in {}s'.format( status, len(train), i, ( loss / count ), ( hit / count ), ( time.time() - start ) ) )
                
            print( 'finished epoch {} with loss {} in {}s'.format( i, ( loss / count ), ( time.time() - start ) ) )
            
         
    def init_model(self, train, std=0.01):
        
        self.I = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I' )
        self.S = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='S' )
        
        self.B = theano.shared( np.random.normal(0, std, size=(self.num_items) ).astype( self.floatX ), name='B' )
        
#         self.H = theano.shared( np.random.normal(0, std, size=(self.num_items) ).astype( self.floatX ), name='H' )
#         self.h_t = theano.shared( np.zeros( self.num_items ).astype( self.floatX ), name='h_t' )
        
        self._generate_train_model_function()
        self._generate_train_model_batch_function()
        self._generate_predict_function()
    
    def init_items(self, train):
        
        index_item = train.columns.get_loc( self.item_key )
                
        for row in train.itertuples(index=False):
            
            ci = row[index_item]
            
            if ci in self.item_map:
                citem = self.item_map[ci]
            else: 
                self.item_map[ci] = self.item_count
                self.item_list[self.item_count] = ci
                self.item_count = self.item_count + 1                  
    
    def _generate_train_model_function(self):
        
        s = T.vector('s', dtype=self.floatX)
        y = T.scalar('y', dtype=self.intX)
        n = T.scalar('n', dtype=self.intX)
        
        #h_t_new = self.H * self.h_t + self.W * s
        #s_t_new = self.V * h_t_new
        se = T.dot( self.S.T, s ) 
        #tmp = T.dot( self.SM.T, s_t_new ) 
        predy = T.dot( self.I[y], se.T ) + self.B[y]
        predn = T.dot( self.I[n], se.T ) + self.B[n]
        
        obj = ( T.log( T.nnet.sigmoid( predy - predn ) ) 
                        - self.regularization * (self.S[y] ** 2).sum()
                        - self.regularization * (self.I[y] ** 2).sum()
                        - self.regularization * (self.S[n] ** 2).sum()
                        - self.regularization * (self.I[n] ** 2).sum()
                        - self.regularization * (self.B[y] ** 2).sum()
                        - self.regularization * (self.B[n] ** 2).sum() )
        
        cost = - obj

        updates = getattr(self, self.learn)(cost, [self.S,self.I,self.B], self.learning_rate)

        self.train_model = theano.function(inputs=[s, y, n], outputs=cost, updates=updates )
    
    def _generate_train_model_batch_function(self):
        
        s = T.matrix('s', dtype=self.floatX)
        y = T.vector('y', dtype=self.intX)
        n = T.vector('n', dtype=self.intX)
        
        #h_t_new = self.H * self.h_t + self.W * s
        #s_t_new = self.V * h_t_new
        se = T.dot( self.S.T, s.T )
        #tmp = T.dot( self.SM.T, s_t_new ) 
        predy = T.dot( self.I[y], se ).diagonal() + self.B[y]
        predn = T.dot( self.I[n], se ).diagonal() + self.B[n]
        
        obj = T.sum( ( T.log( T.nnet.sigmoid( predy - predn ) ) 
                        - self.regularization * (self.S[y] ** 2).sum()
                        - self.regularization * (self.I[y] ** 2).sum(axis=1)
                        - self.regularization * (self.S[n] ** 2).sum(axis=1)
                        - self.regularization * (self.I[n] ** 2).sum(axis=1)
                        - self.regularization * (self.B[y] ** 2)
                        - self.regularization * (self.B[n] ** 2) ) ) 
        
        cost = - obj
        
        updates = getattr(self, self.learn)(cost, [self.S,self.I,self.B], self.learning_rate)
#         updates = self.sgd(cost, [self.S,self.I,self.B], self.learning_rate)

        self.train_model_batch = theano.function(inputs=[s, y, n], outputs=cost, updates=updates )
    
    def _generate_predict_function(self):
        
        s = T.vector('s', dtype=self.floatX)
        
#         h_t_new = self.H * self.h_t + self.W * s
#         s_t_new = self.V * h_t_new
#         pred = T.dot( self.SM, T.dot( self.SM.T, s_t_new ) )
        
        se = T.dot( s.T, self.S ) 
        #tmp = T.dot( self.SM.T, s_t_new ) 
        pred = T.dot( self.I, se.T )
        
        #updates = [ (self.h_t, h_t_new) ]
        
        self.predict = theano.function(inputs=[s], outputs=pred ) #, updates=updates )
        
    
    def sgd(self, loss, param_list, learning_rate=0.01):
        
        all_grads = theano.grad(loss, param_list )
        
        updates = []
        
        for p, g in zip(param_list, all_grads):
            updates.append( (p, p - learning_rate * g ) )
        
        return updates
    
    
    def adam(self, loss, param_list, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        http://arxiv.org/pdf/1412.6980v4.pdf
        """
        
        updates = []
        all_grads = theano.grad(loss, param_list)
        alpha = learning_rate
        t = theano.shared(np.float32(1))
        b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(param_list, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                                dtype=self.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                                dtype=self.floatX))
    
            m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
            m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta) )
            
        updates.append((t, t + 1.))
        
        return updates
    
    def adagrad(self, loss, param_list, learning_rate=1.0, epsilon=1e-6):
        
        updates = []
        all_grads = theano.grad(loss, param_list)
        
        for param, grad in zip(param_list, all_grads):
            value = param.get_value( borrow=True )
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            
            updates.append( ( accu, accu_new ) )
            updates.append( ( param, param - (learning_rate * grad / T.sqrt(accu_new + epsilon) ) ) )
            
        return updates
      
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0 ):
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
                
        if( self.session != session_id ): #new session      
                
            self.session = session_id
            self.session_items = np.zeros(self.num_items, dtype=np.float32)
            self.session_count = 0
            #self.h_t.set_value( np.zeros( (self.num_items) ).astype( self.floatX ) , borrow=True)
        
        if type == 'view':
            self.session_count += 1
            self.session_items[ self.item_map[input_item_id] ] = self.session_count
        
        if skip:
            return
         
        predictions = self.predict( self.session_items / self.session_count )
        series = pd.Series(data=predictions, index=self.item_list)
        series = series[predict_for_item_ids]
        
        return series 
   