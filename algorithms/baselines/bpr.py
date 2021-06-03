import numpy as np
import pandas as pd

class BPR:
    '''
    Code based on work by Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009.

    BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')

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
    def __init__(self, n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None

    def init(self, data):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)
    
    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)
    
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init(data)
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            print(it, np.mean(c))
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view',
                     timestamp=0):
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
        iidx = self.itemidmap[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.itemidmap[predict_for_item_ids]
        return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_item_ids)
             
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
