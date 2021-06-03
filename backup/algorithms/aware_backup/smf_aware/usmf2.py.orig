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
    SessionMatrixFactorization( factors=100, batch=50, learn='adagrad_sub', learning_rate=0.001, momentum=0.0, regularization=0.0001, dropout=0.0, skip=0.0, samples=2048, activation='linear', objective='bpr_max_org', epochs=10, shuffle=3, last_n_days=None, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' )

    Parameters
    -----------
    factors : int
        Number of latent factors. (Default value: 100)
    batch : int
        Batch size for the training process. (Default value: 500)
    learn : string
        Learning method (adagrad_sub, adam, sgd). (default: adagrad_sub)
    momentum : float
        Momentum for adagrad and adam. (default: 0.0)
    regularization : float
        Regulariuation factor for the objective. (default=0.0001)
    dropout : float
        Share of items that are randomly discarded from the current session while training. (default: 0.0)
    skip : float
        Probability that an item is skiped and the next one is used as the positive example. (default: 0.0)
    samples : int
        Number of items that are sampled as negative examples.
    activation : string
        Final activation function (linear, sigmoid, uf_sigmoid, hard_sigmoid, relu, softmax, softsign, softplus, tanh ). (default: 'linear')
	objective : string
        Header of the session ID column in the input file (bpr_max, top1_max, bpr, top1). (default: 'bpr_max_org')
	epochs : int
        Number of training epochs. (default: 10)
	shuffle : int
        Shuffle the order of the sessions for the first N epochs. (default: -1)
	last_n_days : int
        Only use the last n days for the training process. (default: None)
	item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    session_key : string
        Header of the session ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the time column in the input file. (default: 'Time')
    '''
    
    def __init__( self, factors=100, batch=32, learn='adagrad_sub', learning_rate=0.01, momentum=0.0, regularization=0.5, dropout=0.0, skip=0, samples=2048, activation='linear', objective='bpr_max', epochs=10, shuffle=-1, last_n_days=None, session_key = 'SessionId', item_key= 'ItemId', user_key= 'UserId', time_key= 'Time' ):
       
        self.factors = factors
        self.batch = batch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.learn = learn
        self.regularization = regularization
        self.samples = samples
        self.dropout = dropout
        self.skip = skip
        self.shuffle = shuffle
        self.epochs = epochs
        self.activation = activation
        self.objective = objective
        self.last_n_days = last_n_days
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
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
        self.user_map = dict()
        self.user_count = 0
        
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
        
        start = time.time()
        
        self.unique_items = data[self.item_key].unique().astype( self.intX )
        
        self.num_items = data[self.item_key].nunique()
        self.num_users = data[self.user_key].nunique()
        self.itemmap = pd.Series( data=np.arange(self.num_items), index=data[self.item_key].unique() ).astype( self.intX )
        self.usermap = pd.Series( data=np.arange(self.num_users), index=data[self.user_key].unique() ).astype( self.intX )

        #self.unique_users = data[self.user_key].unique().astype( self.intX )
        #self.num_users = data[self.user_key].nunique()
        #self.user_list = np.zeros( self.num_users )
        
        #self.init_items_users(data)
        print( 'finished init item and user map in {}'.format(  ( time.time() - start ) ) )
        
        if self.last_n_days != None:
            
            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]
        
        else: 
            train = data
        
        start = time.time()
                
        self.num_sessions = train[self.session_key].nunique()
        
        train = pd.merge(train, pd.DataFrame({self.item_key:self.itemmap.index, 'ItemIdx':self.itemmap[self.itemmap.index].values}), on=self.item_key, how='inner')
        train = pd.merge(train, pd.DataFrame({self.user_key:self.usermap.index, 'UserIdx':self.usermap[self.usermap.index].values}), on=self.user_key, how='inner')
        train.sort_values([self.session_key, self.time_key], inplace=True)
        
        self.init_model( train )
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
        
        start = time.time()
        
        avg_time = 0
        avg_count = 0
        
        itemvals = train.ItemIdx.values
        uservals = train.UserIdx.values
           
        for j in range( self.epochs ):
                
            loss = 0
            count = 0
            
            batch_size = set(range(self.batch))
            reset_set = set(range(self.batch))
            
            finished = False
            next_sidx = len(batch_size)
            sidx = np.arange(self.batch)
            spos = np.ones( self.batch ).astype( self.intX )
            
            svec = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
            smat = np.zeros( (self.batch, self.num_items) ).astype( self.floatX )
            
            session_offset = np.r_[0, train.groupby([self.session_key], sort=False).size().cumsum()].astype(self.intX)
            itemss = set()
                             
            while not finished:
                
                #ran = np.random.random_sample()
                #itemss = set()
                tmp = session_offset[ sidx ] + spos
                items = itemvals[ tmp - 1 ]
                targets = itemvals[ tmp ]
                users = uservals[ tmp ]
                
                #svec[ np.arange(self.batch) , items ] = spos
                svec[ np.arange(self.batch) , items ] = 1
                for i in range(self.batch):
                    smat[i] = svec[i] / spos[i]
                
                #smat[:,:] = (svec.T / spos).T
                itemss.update( items )
                        
                spos = spos + 1
                
                itemsl = list(itemss)
                if self.samples > 0:
                    #additional = samples
                    additional = np.random.randint(self.num_items, size=self.samples).astype( self.intX )
                    stmp = time.time()
                    
                    loss += self.train_model_batch( smat, items, users, np.hstack( [targets, additional] ), itemsl )
                    avg_time += (time.time() - stmp)
                    avg_count += 1
                else:
                    loss +=  self.train_model_batch( smat, items, users, targets, itemsl )
                
#                 loss = 0.5

                if np.isnan(loss):
                    print(str(j) + ': NaN error!')
                    self.error_during_train = True
                    return
                
                count += self.batch
                
                session_change = np.arange( self.batch )[ (session_offset[ sidx ] + spos) >= (session_offset[ sidx+1 ]) ]
                
                for i in session_change:
                    
                    if next_sidx < len( session_offset ) - 1 :
                        spos[i] = 1
                        sidx[i] = next_sidx
                        svec[i].fill(0)
                        #svec[i,:] = 0
                        next_sidx += 1
                        reset_set -= set([i])
                    else:
                        spos[i] -= 1
                        batch_size -= set([i])
                        finished = True
                        break
                    
                if len(batch_size) == 0:
                    finished = True
                    
                if len(reset_set) == 0:
                    reset_set = set(range(self.batch))
                    itemss = set()
                
#                 if count % 10000 == 0 :
#                     print( 'finished {} of {} in epoch {} with loss {} in {}s'.format( count, len(train), j, ( loss / count ), ( time.time() - start ) ) )
                
            print( 'finished epoch {} with loss {} in {}s'.format( j, ( loss / count ), ( time.time() - start ) ) )
            
        print( 'avg_time_fact: ',( avg_time / avg_count ) )
        print( 'time fact: ',( avg_time ) )
        print( 'count fact: ',( avg_count ) )
        #self.train_model_batch.profile.summary()
        
    def init_model(self, train, std=0.01):
        
        self.I = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I', borrow=True )
        self.S = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='S', borrow=True )
        self.U = theano.shared( np.random.normal(0, std, size=(self.num_users, self.factors) ).astype( self.floatX ), name='U', borrow=True )
        
        self.I1 = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I1', borrow=True )
        self.I2 = theano.shared( np.random.normal(0, std, size=(self.num_items, self.factors) ).astype( self.floatX ), name='I2', borrow=True )

        self.BS = theano.shared( np.random.normal(0, std, size=(self.num_items,1) ).astype( self.floatX ), name='BS', borrow=True )
        self.BU = theano.shared( np.random.normal(0, std, size=(self.num_items,1) ).astype( self.floatX ), name='BS', borrow=True )
        self.BI = theano.shared( np.random.normal(0, std, size=(self.num_items,1) ).astype( self.floatX ), name='BI', borrow=True )
        
        self.hack_matrix = np.ones((self.batch, self.batch + self.samples), dtype=self.floatX)
        np.fill_diagonal(self.hack_matrix, 0)
        self.hack_matrix = theano.shared(self.hack_matrix, borrow=True)
        
        self._generate_train_model_batch_function()
        self._generate_predict_function()
        self._generate_predict_batch_function()
         
    
    def init_items_users(self, train):
        
        index_item = train.columns.get_loc( self.item_key )
        index_user = train.columns.get_loc( self.user_key )
                
        for row in train.itertuples(index=False):
            
            ci = row[index_item]
            cu = row[index_user]
            
            if not ci in self.item_map: 
                self.item_map[ci] = self.item_count
                self.item_list[self.item_count] = ci
                self.item_count = self.item_count + 1
            
            if not cu in self.user_map: 
                self.user_map[cu] = self.user_count
                self.user_list[self.user_count] = ci
                self.user_count += 1
    
    def init_sessions(self, train):
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_user = train.columns.get_loc( self.user_key )  
                                            
        self.sessions = []
        self.session_map = {}
        self.session_user = {}
        
        train.sort_values( [self.session_key,self.time_key], inplace=True )
        
        prev_session = -1
        
        for row in train.itertuples(index=False):
            
            item = self.item_map[ row[index_item] ]
            session = row[index_session]
            
            if prev_session != session: 
                self.sessions.append(session)
                self.session_map[session] = []
                self.session_user[session] = self.user_map[ row[index_user] ]
            
            self.session_map[session].append(item)
            prev_session = session
    
    def _generate_train_model_batch_function(self):
        
        s = T.matrix('s', dtype=self.floatX)
        u = T.vector('u', dtype=self.intX)
        i = T.vector('i', dtype=self.intX)
        y = T.vector('y', dtype=self.intX)
        items = T.vector('items', dtype=self.intX)
        
        Sit = self.S[items]
        sit = s.T[items]
        
        Uu = self.U[u]
        
        Iy = self.I[y]
        
        BSy = self.BS[y]
        BUy = self.BU[y]
        BIy = self.BI[y]
        
        I1i = self.I1[i]
        I2y = self.I2[y]
        
        predU =  T.dot( Iy, Uu.T ).T + BUy.flatten()
        
        se = T.dot( Sit.T, sit )
        predS =  T.dot( Iy, se ).T + BSy.flatten()
        
        predI = T.dot( I1i, I2y.T ) + BIy.flatten()
        
        pred = predS + predI + predU
        pred = getattr(self, self.activation )( pred )
        
        cost = getattr(self, self.objective )( pred, y )
        
        #updates = getattr(self, self.learn)(cost, [self.U,self.S,self.I,self.IC,self.BI,self.BS], self.learning_rate)
        updates = getattr(self, self.learn)(cost, [self.U,self.S,self.I,self.I1,self.I2,self.BI,self.BS], [Uu,Sit,Iy,I1i,I2y,BIy,BSy],[u,items,y,i,y,y,y], self.learning_rate, momentum=self.momentum)
        
        self.train_model_batch = theano.function(inputs=[s, i, u, y, items], outputs=cost, updates=updates  )
    
    def _generate_predict_function(self):
        
        s = T.vector('s', dtype=self.floatX)
        u = T.scalar('u', dtype=self.intX)
        i = T.scalar('i', dtype=self.intX)
        
        se = T.dot( self.S.T, s.T )
        
        predS = T.dot( self.I, se ).T + self.BS.flatten()
        predU = T.dot( self.I, self.U[u].T ).T + self.BU.flatten()
        predI = T.dot( self.I1[i], self.I2.T ) + self.BI.flatten()
        
        pred = predS + predI + predU
        pred = getattr(self, self.activation )( pred )
        
        self.predict = theano.function(inputs=[s, i, u], outputs=pred )
    
    def _generate_predict_batch_function(self):
        
        s = T.matrix('s', dtype=self.floatX)
        u = T.vector('u', dtype=self.intX)
        i = T.vector('i', dtype=self.intX)
        
        se = T.dot( self.S.T, s.T )
        
        predS = T.dot( self.I, se ).T + self.BS
        predU = T.dot( self.I, self.U[u].T ).T + self.BU
        predI = T.dot( self.I1[i], self.I2.T ) + self.BI
        
        pred = predS + predI + predU
        
        pred = getattr(self, self.activation )( pred )
        
        self.predict_batch = theano.function(inputs=[s, i, u], outputs=pred ) #, updates=updates )
        
        
    def bpr_old(self, predy, y ):
        ytrue = predy.diagonal()
        obj = T.sum( ( T.log( T.nnet.sigmoid( ytrue - predy.T ) ) 
                        - self.regularization * (self.S[y] ** 2).sum(axis=1)
                        - self.regularization * (self.I[y] ** 2).sum(axis=1)
                        - self.regularization * (self.IC[y] ** 2).sum(axis=1)
                        - self.regularization * (self.BI[y] ** 2)
                        - self.regularization * (self.BS[y] ** 2) ) ) 
        return -obj
    
    def bpr(self, pred_mat, y ):
        ytrue = pred_mat.T.diagonal()
        obj = -T.sum( T.log( T.nnet.sigmoid( ytrue - pred_mat ) ) )
        return obj
    
    def bpr_max_old(self, pred_mat, y):
        loss=0.5
        softmax_scores = self.softmax_neg(pred_mat.T).T
        return T.cast(T.mean(-T.log(T.sum(T.nnet.sigmoid(T.diag(pred_mat.T)-pred_mat)*softmax_scores, axis=0)+1e-24)+loss*T.sum((pred_mat**2)*softmax_scores, axis=0)), self.floatX)
    
    def bpr_max(self, pred_mat, y):
        softmax_scores = self.softmax_neg(pred_mat).T
        return T.cast(T.mean(-T.log(T.sum(T.nnet.sigmoid(T.diag(pred_mat)-pred_mat.T)*softmax_scores, axis=0)+1e-24)+self.regularization*T.sum((pred_mat.T**2)*softmax_scores, axis=0)), self.floatX)
    
    
    def bpr_mean(self, pred_mat, y ):
        ytrue = pred_mat.T.diagonal()
        obj = -T.mean( T.log( T.nnet.sigmoid( ytrue - pred_mat ) ) )
        return obj
    
    def top1(self, predy, y ):
        ytrue = predy.diagonal()
        obj = T.mean( T.log( T.nnet.sigmoid( -ytrue + predy.T ) ) 
                       - self.regularization * (self.S[y] ** 2).sum(axis=1)
                        - self.regularization * (self.I[y] ** 2).sum(axis=1)
                        - self.regularization * (self.IC[y] ** 2).sum(axis=1)
                        - self.regularization * (self.BI[y] ** 2)
                        - self.regularization * (self.BS[y] ** 2) )
        return obj
    
    def top1_2(self, predy, y):
        predy = predy.T
        obj = T.mean( T.nnet.sigmoid(-T.diag(predy)+predy.T)+T.nnet.sigmoid(predy.T**2) )
        return obj
    
    def top1_max(self, pred_mat, y):
        softmax_scores = self.softmax_neg(pred_mat).T    
        tmp = softmax_scores*(T.nnet.sigmoid(-T.diag(pred_mat)+pred_mat.T)+T.nnet.sigmoid(pred_mat.T**2))      
        return T.mean(T.sum(tmp, axis=0))
    
    def cross_entropy(self, pred_mat, y ):
        obj = T.mean( -T.log( pred_mat.diagonal() + 1e-24 ) )
        return obj
    
    
    def softmax_neg(self, X):
        if hasattr(self, 'hack_matrix'):
            X = X * self.hack_matrix
            e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x')) * self.hack_matrix
        else:
            e_x = T.fill_diagonal(T.exp(X - X.max(axis=1).dimshuffle(0, 'x')), 0)
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    
    
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
    
    def adagrad_sub(self, loss, param_list, subparam_list, idx, learning_rate=1.0, epsilon=1e-6, momentum=0.0 ):
        
        updates = []

        all_grads = theano.grad(loss, subparam_list)
        
        for i in range(len(all_grads)):
            
            grad = all_grads[i]
            param = param_list[i]
            index = idx[i]
            subparam = subparam_list[i]
            
            accu = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
 
            accu_s = accu[index]
            accu_new = accu_s + grad ** 2
            updates.append( ( accu, T.set_subtensor(accu_s, accu_new) ) )
            
            delta = learning_rate * grad / T.sqrt(accu_new + epsilon)
            
            if momentum > 0:
                velocity = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
                vs = velocity[index]
                velocity2 = momentum * vs - delta
                updates.append( ( velocity, T.set_subtensor(vs, velocity2) ) )
                updates.append( ( param, T.inc_subtensor(subparam, velocity2 ) ) )
            else:
                updates.append( ( param, T.inc_subtensor(subparam, - delta ) ) )
            
        return updates
    
    def linear(self, param):
        return param
    
    def sigmoid(self, param):
        return T.nnet.sigmoid( param )
    
    def uf_sigmoid(self, param):
        return T.nnet.ultra_fast_sigmoid( param )
    
    def hard_sigmoid(self, param):
        return T.nnet.hard_sigmoid( param )
    
    def relu(self, param):
        return T.nnet.relu( param )
    
    def softmax(self, param):
        return T.nnet.softmax( param )
    
    def softsign(self, param):
        return T.nnet.softsign( param )
    
    def softplus(self, param):
        return T.nnet.softplus( param )
    
    def tanh(self, param):
        return T.tanh( param )
     
    def predict_next( self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0 ):
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
        
        if type == 'view':
            self.session_count += 1
            self.session_items[ self.itemmap[input_item_id] ] = self.session_count
            #self.session_items[ self.itemmap[input_item_id] ] = 1
        
        if skip:
            return
         
        predictions = self.predict( self.session_items / self.session_count, self.itemmap[input_item_id], self.usermap[input_user_id] )
        #predictions = self.predict( self.session_items, self.itemmap[input_item_id], self.usermap[input_user_id] )
        series = pd.Series(data=predictions, index=self.itemmap.index)
        series = series[predict_for_item_ids]
                
        return series
    
    def clear(self):
        self.I.set_value([[]])
        self.S.set_value([[]])
        self.I1.set_value([[]])
        self.I2.set_value([[]])
        self.BS.set_value([[]])
        self.BI.set_value([[]])
        self.hack_matrix.set_value([[]])
        
