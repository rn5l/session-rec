import math
import numpy as np
from algorithms.NCFS.models.cross_sess_model import CrossSessRS
import pandas as pd
import keras

class NCFS:

    def __init__(self, window_sz = 8, max_nb_his_sess = 2, mini_batch_sz=200, neg_samples=100, max_epoch = 10, max_session_len = None, embeding_len = 100, dropout=0, att_alpha = 0.01,
                 session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId'):  # max_session_len = 20 # todo: check the speed of training/test with limited max session length, e.g., 20
        print("init")
        self.window_sz = window_sz  # half context window size # number of most recent choices around a target item, for modeling the intra-context
        self.max_nb_his_sess = max_nb_his_sess  # max number of history session # number of negative samples # number of historical sessions as the inter-session context
        self.mini_batch_sz = mini_batch_sz  # batch size for training
        self.neg_samples = neg_samples  # negative sampling, to approximate the softmax computation over all items. we randomly draw a small set of items as the N noise samples
        self.test_batch_sz = 1  # batch size for test
        self.max_epoch = max_epoch  # max epoch
        self.max_session_len = max_session_len  # max number of items in a session
        self.embeding_len = embeding_len
        self.dropout = dropout
        self.att_alpha = att_alpha
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key
        # for prediction phase
        self.position = 0
        self.session = -1
        self.session_items = []

    # Renumber items to start from 1
    def prepare_item_maps(self, train_data):
        item_dict = {}
        reversed_item_dict = {}
        item_ctr = 1
        for index, row in train_data.iterrows():
            itemId = row[self.item_key]
            if not itemId in item_dict:
                item_dict[itemId] = item_ctr
                reversed_item_dict[item_ctr] = itemId
                item_ctr += 1
        return item_dict, reversed_item_dict

    def init_model(self, train_data, test_data):

        if self.max_session_len is None:
            max_session_length_train = train_data.groupby(self.session_key).size().max()
            max_session_length_test = test_data.groupby(self.session_key).size().max()
            self.max_session_len = max(max_session_length_train, max_session_length_test)
            print("max session lenght:" + str(self.max_session_len))


        self.nb_item = len(self.item_dict)

        self.sessionSet = dict()  # dictionary: {userId: [(Session1)[ItemId, ItemId, ItemId],(Session2)[ItemId, ItemId, ItemId, ItemId]]}
        session_prev = -1
        user_prev = -1
        for index, row in train_data.iterrows():
            userId = row[self.user_key]
            sessionId =  row[self.session_key]
            itemId = self.item_dict[row[self.item_key]]
            if not userId in self.sessionSet.keys():
                if user_prev != -1:
                    self.sessionSet[user_prev].append(sess)
                self.sessionSet[userId] = []
                sess = []
            elif sessionId != session_prev:
                if session_prev != -1:
                    self.sessionSet[userId].append(sess)
                sess = []
            sess.append(itemId)
            user_prev = userId
            session_prev = sessionId

        self.nb_trn_spl = 0  # number of test set cases for training set
        for k, v in self.sessionSet.items():
            for i in range(1, len(v)):
                self.nb_trn_spl += len(v[i])  # number of training set cases

        self.nb_batch = math.ceil(self.nb_trn_spl / self.mini_batch_sz)  # number of batches for training

        crossRS = CrossSessRS(num_items=self.nb_item, neg_samples=self.neg_samples, embedding_len=self.embeding_len,
                              ctx_len=2 * self.window_sz,
                              max_sess_len=self.max_session_len, max_nb_sess=self.max_nb_his_sess,
                              dropout = self.dropout, att_alpha=self.att_alpha)  # embedding_len = 100 units for the item embeddings

        self.trainModel = crossRS.train_model  # train_model (from CrossSessRS)
        self.trainModel.summary()
        self.trainModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

        self.pred_model = crossRS.predict_model  # predict_model (from CrossSessRS)


    def fit(self, train_data, test_data, valid_data=None, retrain=False, sample_store=10000000, patience=3, margin=1.003,
            save_to=None, load_from=None):
        '''
        Trains the network.

        Parameters
        --------
        train_data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        valid_data: pandas.DataFrame
            Validation data. If not none, it enables early stopping.
             Contains the transactions in the same format as in train_data, and it is used exclusively to compute the loss after each training iteration over train_data.
        retrain : boolean
            If False, do normal train. If True, do additional train (weights from previous trainings are kept as the initial network) (default: False)
        sample_store : int
            If additional negative samples are used (n_sample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).
        patience: int
            Patience of the early stopping procedure. Number of iterations with not decreasing validation loss before terminating the training procedure
        margin: float
            Margin of early stopping. Percentage improvement over the current best validation loss to do not incur into a patience penalty
        save_to: string
            Path where to save the state of the best model resulting from training.
            If early stopping is enabled, saves the model with the lowest validation loss. Otherwise, saves the model corresponding to the last iteration.
        load_from: string
            Path from where to load the state of a previously saved model.
        '''

        print("fit")
        self.item_dict, self.reversed_item_dict = self.prepare_item_maps(train_data)
        self.init_model(train_data, test_data)
        spl_idx = 0
        spl_cnt = 0
        for ep in range(self.max_epoch):
            # Training
            curr_batch = 0
            batch_sz = min(self.mini_batch_sz, self.nb_trn_spl)
            his_input = np.zeros([batch_sz, self.max_nb_his_sess, self.max_session_len], dtype=np.int32)
            sess_input = np.zeros([batch_sz, 2 * self.window_sz], dtype=np.int32)
            target_input = np.zeros([batch_sz, self.neg_samples + 1], dtype=np.int32)
            labels = np.zeros((batch_sz,), dtype=np.int32)
            for user_id, v in self.sessionSet.items(): # v is a list of sessions' of the current user
                for i in range(1, len(v)):
                    sess = v[i]  # sess includes the itemId s of the current session
                    for c in range(len(sess)): # for each item in the session
                        target_input[spl_idx, 0] = sess[c]  # store itemId in target_input(ndarray) in (row: spl_idx, column: 0)
                        target_input[spl_idx, 1:] = np.random.randint(0, self.nb_item, self.neg_samples)  # in other columns (from 1 to 100) store random itemIds in range (0, nb_item) with size neg_samples
                        ctx_sess = np.concatenate([sess[max(0,c-self.window_sz):c], sess[(c+1):min(c+1+self.window_sz,len(sess))]])
                        sess_input[spl_idx, -len(ctx_sess):]  = ctx_sess
                        for j in range(min(i, self.max_nb_his_sess)):
                            self.his_sess = v[i - 1 - j]
                            if len(self.his_sess) > self.max_session_len:
                                self.his_sess = self.his_sess[:self.max_session_len]
                            his_input[spl_idx, -(j + 1), :len(self.his_sess)] = self.his_sess
                        spl_idx += 1
                        if spl_idx == batch_sz:
                            spl_idx = 0
                            curr_batch += 1
                            spl_cnt += batch_sz
                            batch_sz = min(self.mini_batch_sz, self.nb_trn_spl - spl_cnt)
                            metrics = self.trainModel.train_on_batch(x=[sess_input, his_input, target_input], y=labels)

                            if curr_batch % 100 == 0 or curr_batch == self.nb_batch:
                                print('training %d/%d, acc: %g' % (curr_batch, self.nb_batch, metrics[1]))

                            his_input = np.zeros([batch_sz, self.max_nb_his_sess, self.max_session_len], dtype=np.int32)
                            sess_input = np.zeros([batch_sz, 2 * self.window_sz], dtype=np.int32)
                            target_input = np.zeros([batch_sz, self.neg_samples + 1], dtype=np.int32)
                            labels = np.zeros((batch_sz,), dtype=np.int32)
            spl_cnt = 0



    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()

        item_id_dic = self.item_dict[input_item_id]
        self.session_items.append(item_id_dic)


        batch_sz = 1
        his_input = np.zeros([batch_sz, self.max_nb_his_sess, self.max_session_len],dtype=np.int32)  # history(ies) # <class 'tuple'>: (50, 2, 20): batch_size=50, max_nb_his_sess=2, max-session_length=20
        sess_input = np.zeros([batch_sz, 2 * self.window_sz], dtype=np.int32)  # context(s) of the session
        user_sess = self.sessionSet[input_user_id]  # k: user_id - retrieve sessions of the corresponding user

        c = len(self.session_items)
        ctx_sess = np.array(self.session_items[max(0, c - self.window_sz):c])  # (corrected!) take "window_sz" items before the current item as the session's context
        sess_input[0, -len(ctx_sess):] = ctx_sess  # update the context of the session
        for j in range(min(len(user_sess) - 1,self.max_nb_his_sess)):  # for every session of the user from the history (min of his session and the number of sessions should be considered)
            self.his_sess = user_sess[len(user_sess) - 1 - j]  # take the session from the history
            if len(self.his_sess) > self.max_session_len:  # if the number of items in the session (from the history) is greater than "max_session_len"
                self.his_sess = self.his_sess[:self.max_session_len]  # just take the last "max_session_len" items
            his_input[0, -(j + 1), :len(self.his_sess)] = self.his_sess  # add items of the user's session to the "his_input" (history)
        scores = self.pred_model.predict_on_batch([sess_input, his_input, np.tile(np.arange(0, self.nb_item + 1), (batch_sz,1))])  # including 50 rows (=batch_size). Each row contains the score for all items (the column headers are "item_id"s and the values are the score of the corresponding item_id) #  numpy.tile(A, reps) -> Construct an array by repeating A the number of times given by reps.
        self.predicted_item_ids = []
        scores = scores[0]
        for idx in range(1, len(scores)):  # because in item_dic, indexes start from 1 (not 0)
            self.predicted_item_ids.append(
                int(self.reversed_item_dict[idx]))
        series = pd.Series(data=scores[1:], index=self.predicted_item_ids)
        return series


    def clear(self):
        keras.backend.clear_session()


    def support_users(self):
        '''
          whether it is a session-based or session-aware algorithm
          (if returns True, method "predict_with_training_data" must be defined as well)

          Parameters
          --------

          Returns
          --------
          True : if it is session-aware
          False : if it is session-based
        '''
        return True

    def predict_with_training_data(self):
        '''
            (this method must be defined if "support_users is True")
            whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

            Parameters
            --------

            Returns
            --------
            True : e.g. hgru4rec
            False : e.g. uvsknn
            '''
        return False