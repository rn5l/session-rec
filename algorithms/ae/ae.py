# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
import bottleneck as bn
from algorithms.ae.helper.dae import MultiDAE
from algorithms.ae.helper.vae import MultiVAE
import sys
import time

class AutoEncoder:
    '''
    AutoEncoder(layers = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
    
    Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of the session is modelled as the average of the feature vectors of the items that have occurred in it so far.
        
    Parameters
    --------
    n_factor : int
        The number of features in a feature vector. (Default value: 100)
    n_iterations : int
        The number of epoch for training. (Default value: 10)
    learning_rate : float
        Learning rate. (Default value: 0.01)
    lambda_session : float
        Regularization for session features. (Default value: 0.0)
    lambda_item : float
        Regularization for item features. (Default value: 0.0)
    sigma : float
        The width of the initialization. (Default value: 0.05)
    init_normal : boolean
        Whether to use uniform or normal distribution based initialization.
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    
    '''
    def __init__(self, layers=[200,600], epochs = 10, lr = 0.05, reg=0.00001, algo='vae', session_key = 'SessionId', item_key = 'ItemId', folder=''):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.folder = folder + 'aemodel'
        
        self.algo = algo
        self.session_key = session_key
        self.item_key = item_key
        
        self.session_items = []
        self.session = -1
        
        self.tfsess = None
        

    def fit( self, data ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
                    
        data = self.filter_data(data,min_uc=5,min_sc=0)    
            
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
        self.predvec = np.zeros( self.n_items )
        
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        data = pd.merge(data, pd.DataFrame({self.item_key:self.itemidmap.index, 'iid':self.itemidmap[self.itemidmap.index].values}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:self.useridmap.index, 'uid':self.useridmap[self.useridmap.index].values}), on=self.session_key, how='inner')
        
        
        #data = self.filter_data(data)
                
        n_kept_users = int( self.n_sessions * 0.9 )
        
        data_val = data[ data.uid > n_kept_users ]
        data = data[ data.uid <= n_kept_users ]
        
        print(len(data_val))
        print(len(data_val))
        
        ones = np.ones( len(data) )
        col_ind = self.itemidmap[ data[self.item_key].values ]
        row_ind = self.useridmap[ data[self.session_key].values ] 
        mat = sparse.csr_matrix((ones, (row_ind, col_ind)), shape=(self.n_sessions, self.n_items))
        
        data_val_tr, data_val_te = self.split_train_test_proportion( data_val )
                
        data_val_tr['uid'] = self.useridmap[ data_val_tr[self.session_key].values ].values
        data_val_tr['uid'] = data_val_tr['uid'] - data_val_tr['uid'].min() 
        ones = np.ones( len(data_val_tr) )
        col_ind = self.itemidmap[ data_val_tr[self.item_key].values ]
        row_ind = data_val_tr.uid.values
        mat_val_tr = sparse.csr_matrix((ones, (row_ind, col_ind)) , shape=( data_val_tr[self.session_key].nunique() , self.n_items))
        
        data_val_te['uid'] = self.useridmap[ data_val_te[self.session_key].values ].values
        print( data_val_te['uid'].max() )
        print( data_val_te['uid'].min() )
        data_val_te['uid'] = data_val_te['uid'] - data_val_te['uid'].min() 
        ones = np.ones( len(data_val_te) )
        col_ind = self.itemidmap[ data_val_te[self.item_key].values ]
        row_ind = data_val_te.uid.values
        print( data_val_te['uid'].max() )
        print( data_val_te['uid'].min() )
        print( data_val_te[self.session_key].nunique() )
        mat_val_te = sparse.csr_matrix((ones, (row_ind, col_ind)), shape=( data_val_te[self.session_key].nunique() , self.n_items))
        
        self.layers = self.layers + [self.n_items]
        
        if self.algo == 'dae':
            ae = MultiDAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )
        elif self.algo == 'vae': 
            ae = MultiVAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )

        self.model = ae
        
        N = mat.shape[0]
        idxlist = np.array( range(N) )
        
        # training batch size
        batch_size = 50
        batches_per_epoch = int(np.ceil(float(N) / batch_size))
        
        N_vad = mat_val_tr.shape[0]
        idxlist_vad = np.array( range(N_vad) )
        
        # validation batch size (since the entire validation set might not fit into GPU memory)
        batch_size_vad = 50
        
        # the total number of gradient updates for annealing
        total_anneal_steps = 200000
        # largest annealing parameter
        anneal_cap = 0.2
        
        saver, logits_var, loss_var, train_op_var, merged_var = ae.build_graph()
        
        ndcg_var = tf.Variable(0.0)
        ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
        ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
        ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
        merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])
        
        summary_writer = tf.summary.FileWriter(self.folder, graph=tf.get_default_graph())

        ndcgs_vad = []

        with tf.Session() as sess:
        
            init = tf.global_variables_initializer()
            sess.run(init)
        
            best_ndcg = -np.inf
        
            update_count = 0.0
            
            tstart = time.time()
            
            for epoch in range(self.epochs):
                np.random.shuffle(idxlist)
                # train for one epoch
                for bnum, st_idx in enumerate(range(0, N, batch_size)):
                    end_idx = min(st_idx + batch_size, N)
                    X = mat[idxlist[st_idx:end_idx]]
                    
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')           
                    
                    if total_anneal_steps > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = anneal_cap
                    
                    if self.algo == 'dae':
                        feed_dict = {ae.input_ph: X, 
                                     ae.keep_prob_ph: 0.5}  
                    else:
                        feed_dict = {ae.input_ph: X, 
                                 ae.keep_prob_ph: 0.5, 
                                 ae.anneal_ph: anneal,
                                 ae.is_training_ph: 1}
                        
                    sess.run(train_op_var, feed_dict=feed_dict)
        
                    if bnum % 100 == 0:
                        summary_train = sess.run(merged_var, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_train, 
                                                   global_step=epoch * batches_per_epoch + bnum) 
                        
                        print( 'finished {} of {} in epoch {} in {}s'.format( bnum, batches_per_epoch, epoch, ( time.time() - tstart ) ) )
                    
                    update_count += 1
                
                # compute validation NDCG
                ndcg_dist = []
                for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                    end_idx = min(st_idx + batch_size_vad, N_vad)
                    X = mat_val_tr[idxlist_vad[st_idx:end_idx]]
        
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')
                    
                    pred_val = sess.run(logits_var, feed_dict={ae.input_ph: X} )
                    # exclude examples from training and validation (if any)
                    pred_val[X.nonzero()] = -np.inf
                    ndcg_dist.append(self.NDCG_binary_at_k_batch(pred_val, mat_val_te[idxlist_vad[st_idx:end_idx]]))
                                
                ndcg_dist = np.concatenate(ndcg_dist)
                ndcg_ = ndcg_dist.mean()
                ndcgs_vad.append(ndcg_)
                merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
                summary_writer.add_summary(merged_valid_val, epoch)
        
                # update the best model (if necessary)
                if ndcg_ > best_ndcg:
                    saver.save(sess, '{}/model'.format(self.folder))
                    best_ndcg = ndcg_
                
                print( 'finished epoch {} in {}s with ndcg {}'.format( epoch, ( time.time() - tstart ), ndcg_ ) )
        
        self.predictor = logits_var
        self.predvec = np.zeros( (1, self.n_items) )
        
    def filter_data(self, data, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users. 
        if min_sc > 0:
            itemcount = data[[self.item_key]].groupby(self.item_key).size()
            data = data[data[self.item_key].isin(itemcount.index[itemcount.values >= min_sc])]
        
        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = data[[self.session_key]].groupby(self.session_key).size()
            data = data[data[self.session_key].isin(usercount.index[usercount.values >= min_uc])]
        
        return data
    
    def split_train_test_proportion(self, data, test_prop=0.2):
        
        data_grouped_by_user = data.groupby( self.session_key )
        tr_list, te_list = list(), list()
    
        np.random.seed(98765)
    
        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
    
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
    
                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)
    
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()
        
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        
        return data_tr, data_te
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : int or string
            The session IDs of the event.
        tracks : int list
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        res : pandas.DataFrame
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        if self.tfsess is None:
            tf.reset_default_graph()
            self.tfsess = tf.Session()
            if self.algo == 'dae':
                self.model = MultiDAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )
            elif self.algo == 'vae': 
                self.model = MultiVAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )
            saver, logits_var, _, _, _ = self.model.build_graph()
            
            self.saver = saver
            self.predictor = logits_var
            
            self.saver.restore(self.tfsess, '{}/model'.format(self.folder))
        
        ae = self.model
        logits_var = self.predictor
        
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.predvec[0].fill(0)
        
        if type == 'view':
            self.session_items.append( input_item_id )
            if input_item_id in self.itemidmap:
                self.predvec[0][ self.itemidmap[input_item_id] ] = 1
            
        if skip:
            return
        
        recommendations = self.tfsess.run(logits_var, feed_dict={ae.input_ph: self.predvec})
        
        series = pd.Series( data=recommendations[0], index=self.itemidmap.index )

        return series
    
    
    def NDCG_binary_at_k_batch(self, X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))
    
        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                         for n in heldout_batch.getnnz(axis=1)])
        return DCG / IDCG
        