from _collections import OrderedDict
from _operator import itemgetter
from datetime import datetime as dt
from datetime import timedelta as td
from math import log10
from math import sqrt
import random
import scipy.sparse
import time

from pympler import asizeof
from scipy.sparse.csc import csc_matrix
import theano

import numpy as np
import pandas as pd
import theano.tensor as T


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
    
    RAND = 0
    BATCH = 1
    SESSION = 2
    MIXED = 3
    
    def __init__( self, factors=100, batch=50, sampling=RAND, learn='adam', learning_rate=0.001, regularization=0.0001, activation='linear', objective='bpr_old', epochs=5, last_n_days=None, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
       
        self.factors = factors
        self.batch = batch
        self.learning_rate = learning_rate
        self.learn = learn
        self.regularization = regularization
        self.epochs = epochs
        self.activation = activation
        self.objective = objective
        self.sampling = sampling
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
        
        self.floatX = theano.config.floatX
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
        
        self.unique_items = data[self.item_key].unique().astype( self.intX )
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
        
        self.init_sessions( train )
            
        start = time.time()
        self.init_model( train )
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
        
        
        train_actions = len(train) - train[self.session_key].nunique()
        
        start = time.time()
                
        for j in range( self.epochs ):
            
            loss = 0
            count = 0
            hit = 0
            
            batch_size = set(range(self.batch))
                    
            ipos = np.zeros( self.batch ).astype( self.intX )
            ineg = np.zeros( self.batch ).astype( self.intX )
            
            finished = False
            next_sidx = len(batch_size)
            sidx = np.arange(self.batch)
            spos = np.ones( self.batch ).astype( self.intX )
            svec = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
            smat = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
            
            sneg = {}
            
            
            while not finished:
                
                rand = []
                
                for i in range(self.batch):
                    
                    ipos[i] = self.session_map[ self.sessions[ sidx[i] ] ][ spos[i] ]
                    svec[i][ self.session_map[ self.sessions[ sidx[i] ] ][ spos[i] - 1 ] ] = spos[i]
                    smat[i] = svec[i] / spos[i]
                    
                    spos[i] += 1
                    
                    #if self.sampling == SessionMF.SESSION:
                    if i in sneg and len( sneg[i] ) > 0:
                        ineg[i] = sneg[i][0]
                    else:
                        rand.append(i)
                
                if self.sampling == SessionMF.MIXED:
                    sampling = np.random.randint(3)
                else:
                    sampling = self.sampling
                
                if sampling == SessionMF.RAND:
                    ineg = np.random.randint(self.num_items, size=self.batch)
                elif sampling == SessionMF.BATCH:
                    ineg = np.copy(ipos)
                    np.random.shuffle( ineg )
                elif sampling == SessionMF.SESSION:
                    tmp = np.copy(ipos)
                    np.random.shuffle( tmp )
                    ineg[rand] = tmp[rand]
                
                loss += self.train_model_batch( smat, ipos, ineg )
                count += self.batch
                
                #HITRATE
                preds = self.predict_batch( smat )
                val_pos = preds.T[ipos].T.diagonal()
                hit += ( (preds.T > val_pos).sum(axis=0) < 20 ).sum()
                #HITRATE
                
                for i in range(self.batch):
                    
                    #NEG SAMPLES
                    #if self.sampling == SessionMF.SESSION:
                    candidates = set( np.where( preds[i] > val_pos[i] )[0] )
                    session = self.session_map[ self.sessions[ sidx[i] ] ]
                    to_come = set( session[-(len(session)-spos[i]):] )
                    sneg[i] = list(candidates - to_come)
                    
                    if len( self.session_map[ self.sessions[ sidx[i] ] ] ) == spos[i]: #session end
                        if next_sidx < len( self.sessions ):
                            spos[i] = 1
                            sidx[i] = next_sidx
                            svec[i] = np.zeros( self.num_items ).astype( self.floatX )
                            smat[i] = np.zeros( self.num_items ).astype( self.floatX )
                            sneg[i] = []
                            next_sidx += 1
                        else:
                            spos[i] -= 1
                            batch_size -= set([i])
                    
                    if len(batch_size) == 0:
                        finished = True
                            
                if count % 10000 == 0 :
                    print( 'finished {} of {} in epoch {} with loss {} / hr {} in {}s'.format( count, train_actions, j, ( loss / count ), ( hit / count ), ( time.time() - start ) ) )
                
            print( 'finished epoch {} with loss {} / hr {} in {}s'.format( j, ( loss / count ), ( hit / count ), ( time.time() - start ) ) )
            
         
    def init_model(self, train, std=0.01):
        
        self.I = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I' )
        self.S = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='S' )
        
        self.B = theano.shared( np.random.normal(0, std, size=(self.num_items) ).astype( self.floatX ), name='B' )
        
#         self.H = theano.shared( np.random.normal(0, std, size=(self.num_items) ).astype( self.floatX ), name='H' )
#         self.h_t = theano.shared( np.zeros( self.num_items ).astype( self.floatX ), name='h_t' )
        
        self._generate_train_model_function()
        self._generate_train_model_batch_function()
        self._generate_predict_function()
        self._generate_predict_batch_function()
    
    def init_items(self, train):
        
        index_item = train.columns.get_loc( self.item_key )
                
        for row in train.itertuples(index=False):
            
            ci = row[index_item]
            
            if not ci in self.item_map: 
                self.item_map[ci] = self.item_count
                self.item_list[self.item_count] = ci
                self.item_count = self.item_count + 1                  
    
    def init_sessions(self, train):
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        self.sessions = []
        self.session_map = {}
        
        train.sort_values( [self.session_key,self.time_key], inplace=True )
        
        prev_session = -1
        
        for row in train.itertuples(index=False):
            
            item = self.item_map[ row[index_item] ]
            session = row[index_session]
            
            if prev_session != session: 
                self.sessions.append(session)
                self.session_map[session] = []
            
            self.session_map[session].append(item)
            prev_session = session
        
    
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
        predy = getattr(self, self.activation )( predy )
        predn = T.dot( self.I[n], se ).diagonal() + self.B[n]
        predn = getattr(self, self.activation )( predn )
        
        cost = getattr(self, self.objective )( predy, predn, y, n )
        
        updates = getattr(self, self.learn)(cost, [self.S,self.I,self.B], self.learning_rate)
#         updates = self.sgd(cost, [self.S,self.I,self.B], self.learning_rate)

        self.train_model_batch = theano.function(inputs=[s, y, n], outputs=cost, updates=updates, on_unused_input='ignore' )
    
    def _generate_predict_function(self):
        
        s = T.vector('s', dtype=self.floatX)
        
#         h_t_new = self.H * self.h_t + self.W * s
#         s_t_new = self.V * h_t_new
#         pred = T.dot( self.SM, T.dot( self.SM.T, s_t_new ) )
        
        se = T.dot( s.T, self.S ) 
        #tmp = T.dot( self.SM.T, s_t_new ) 
        pred = T.dot( self.I, se.T ) + self.B
        pred = getattr(self, self.activation )( pred )
        
        #updates = [ (self.h_t, h_t_new) ]
        
        self.predict = theano.function(inputs=[s], outputs=pred ) #, updates=updates )
    
    def _generate_predict_batch_function(self):
        
        s = T.matrix('s', dtype=self.floatX)
        
        #h_t_new = self.H * self.h_t + self.W * s
        #s_t_new = self.V * h_t_new
        se = T.dot( self.S.T, s.T )
        #tmp = T.dot( self.SM.T, s_t_new ) 
        pred = T.dot( self.I, se ).T + self.B
        pred = getattr(self, self.activation )( pred )
        #updates = [ (self.h_t, h_t_new) ]
        
        self.predict_batch = theano.function(inputs=[s], outputs=pred ) #, updates=updates )
        
    def bpr_old(self, predy, predn, y, n ):
        obj = T.sum( ( T.log( T.nnet.sigmoid( predy - predn ) ) 
                        - self.regularization * (self.S[y] ** 2).sum(axis=1)
                        - self.regularization * (self.I[y] ** 2).sum(axis=1)
                        - self.regularization * (self.S[n] ** 2).sum(axis=1)
                        - self.regularization * (self.I[n] ** 2).sum(axis=1)
                        - self.regularization * (self.B[y] ** 2)
                        - self.regularization * (self.B[n] ** 2) ) ) 
        return -obj
    
    def bpr(self, predy, predn, y, n ):
        obj = -T.sum( T.log( T.nnet.sigmoid( predy - predn ) ) )
        return obj
    
    def bpr_mean(self, predy, predn, y, n ):
        obj = -T.mean( T.log( T.nnet.sigmoid( predy - predn ) ) )
        return obj
    
    def top1(self, predy, predn, y, n ):
        obj = T.mean( T.log( T.nnet.sigmoid( predn - predy ) ) 
                        - self.regularization * (self.S[y] ** 2).sum(axis=1)
                        - self.regularization * (self.I[y] ** 2).sum(axis=1)
                        - self.regularization * (self.S[n] ** 2).sum(axis=1)
                        - self.regularization * (self.I[n] ** 2).sum(axis=1)
                        - self.regularization * (self.B[y] ** 2)
                        - self.regularization * (self.B[n] ** 2) )
        return obj
    
    def cross_entropy(self, predy, predn, y, n ):
        obj = T.mean( -T.log( predy + 1e-24 ) )
        return obj
    
    
    def sgd(self, loss, param_list, learning_rate=0.01):
        
        all_grads = theano.grad(loss, param_list )
        
        updates = []
        
        for p, g in zip(param_list, all_grads):
            updates.append( (p, p - learning_rate * g ) )
        
        return updates
    
    
    def adam(self, cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        updates = []
        all_grads = theano.grad(cost, params)
        
        t_prev = theano.shared(np.float32(0.))
    
        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)
    
        t = t_prev + 1
        a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)
    
        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
    
            m_t = beta1*m_prev + (one-beta1)*g_t
            v_t = beta2*v_prev + (one-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)
            
            updates.append( ( m_prev, m_t ) )
            updates.append( ( v_prev, v_t ) )
            updates.append( ( param, param - step ) )
            
        updates.append( ( t_prev, t ) )
        return updates
    
    def adam2(self, loss, param_list, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):

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
    
    def adamax(self, cost, params, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        updates = []
        all_grads = theano.grad(cost, params)
        
        t_prev = theano.shared(np.float32(0.))
    
        one = T.constant(1)
    
        t = t_prev + 1
        a_t = learning_rate/(one-beta1**t)
    
        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
    
            m_t = beta1*m_prev + (one-beta1)*g_t
            u_t = T.maximum(beta2*u_prev, abs(g_t))
            step = a_t*m_t/(u_t + epsilon)
    
            updates.append( ( m_prev, m_t ) )
            updates.append( ( u_prev, u_t ) )
            updates.append( ( param, param - step ) )
            
        updates.append( ( t_prev, t ) )
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
    
    def adadelta(self, cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):

        updates = []
        all_grads = theano.grad(cost, params)
    
        one = T.constant(1)
    
        for param, grad in zip(params, all_grads):
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)
    
            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates.append( ( accu, accu_new ) )
    
            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + epsilon) /
                      T.sqrt(accu_new + epsilon))
            updates.append( ( param, param - learning_rate * update ) )
    
            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates.append( ( delta_accu, delta_accu_new ) )
    
        return updates
    
    def linear(self, param):
        return param
    
    def sigmoid(self, param):
        return T.nnet.sigmoid( param )
    
    def relu(self, param):
        return T.nnet.relu( param )
    
    def softmax(self, param):
        return T.nnet.softmax( param )
    
    def softsign(self, param):
        return T.nnet.softsign( param )
    
    def tanh(self, param):
        return T.tanh( param )
     
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
   