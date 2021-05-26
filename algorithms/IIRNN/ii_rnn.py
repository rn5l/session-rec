# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # will probably be moved to code in TF 1.1. Keep it imported as rnn to make the rest of the code independent of this.
import datetime
import os
import time
import math
import numpy as np
from algorithms.IIRNN.utils_ii_rnn import IIRNNDataHandler
import pandas as pd
# from test_util import Tester

class IIRNN:
    """

    use_last_hidden_state defines whether to use last hidden state or average of embeddings as session representation.
    BATCHSIZE defines the number of sessions in each mini-batch.
    ST_INTERNALSIZE defines the number of nodes in the intra-session RNN layer (ST = Short Term)
    LT_INTERNALSIZE defines the number of nodes in the inter-session RNN layer. These two depends on each other and needs to be the same size as each other and as the embedding size. If you want to use different sizes, you probably need to change the model as well, or at least how session representations are created.
    learning_rate is what you think it is.
    dropout_pkeep is the propability to keep a random node. So setting this value to 1.0 is equivalent to not using dropout.
    MAX_SESSION_REPRESENTATIONS defines the maximum number of recent session representations to consider.
    MAX_EPOCHS defines the maximum number of training epochs before the program terminates. It is no problem to manually terminate the program while training/testing the model and continue later if you have save_best = True. But notice that when you start the program again, it will load and continue training the last saved (best) model. Thus, if you are currently on epoch #40, and the last best model was achieved at epoch #32, then the training will start from epoch #32 again when restarting.
    N_LAYERS defines the number of GRU-layers used in the intra-session RNN layer.
    SEQLEN should be set to the MAX_SESSION_LENGTH - 1 (from preprocessing). This is the length of sessions (with padding), (minus one since the last user interactions is only used as a label for the previous one).
    TOP_K defines the number of items the model produces in each recommendation.
    """

    def __init__(self, embedding_size= 100, max_epoch=20, batch_size=100, max_session_representation=15,
                 learning_rate=0.05, dropout_pkeep=0.0, use_last_hidden_state = True, do_training = True, save_best = True,
                 session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId'):
        self.EMBEDDING_SIZE = embedding_size
        self.ST_INTERNALSIZE = embedding_size # Embedding size
        self.LT_INTERNALSIZE = embedding_size
        self.MAX_EPOCHS = max_epoch
        self.BATCHSIZE = batch_size
        self.MAX_SESSION_REPRESENTATIONS = max_session_representation
        self.learning_rate = learning_rate
        self.dropout_pkeep = dropout_pkeep
        self.use_last_hidden_state = use_last_hidden_state
        self.do_training = do_training
        self.save_best = save_best
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key

        seed = 0
        tf.set_random_seed(seed)

        # self.N_ITEMS = -1

        self.N_LAYERS = 1  # number of layers in the rnn

        # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions.
        # This is important for training, but when running, we can have as long sequences as we want!
        # Just need to keep the hidden state and compute the next action)
        self.SEQLEN = 20 - 1 # self.SEQLEN = 20 - 1 # TODO: must be set correctly?! (seems that it is not used anywhere!)

        self.TOP_K = 50 # in fit() it will be updated, therefore it will return the ranking for all items, not just first 50

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.session_representation = None
        self.user_sess_representation = None


    def init_model(self):
        ##
        ## The model
        ##
        print("Creating model")
        cpu = ['/cpu:0']
        gpu = ['/gpu:0', '/gpu:1']

        # Use (CPU) RAM to hold embeddings. If >10 GB of VRAM available, you can put
        # this there instead, which should reduce runtime
        with tf.device(cpu[0]):
            # Inputs
            self.X = tf.placeholder(tf.int32, [None, None], name='X')  # [ BATCHSIZE, SEQLEN ]
            self.Y_ = tf.placeholder(tf.int32, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]

            # Embeddings. W_embed = all embeddings. X_embed = retrieved embeddings
            # from W_embed, corresponding to the items in the current batch
            W_embed = tf.Variable(tf.random_uniform([self.N_ITEMS, self.EMBEDDING_SIZE], -1.0, 1.0), name='embeddings')
            X_embed = tf.nn.embedding_lookup(W_embed, self.X)  # [BATCHSIZE, SEQLEN, EMBEDDING_SIZE]

        with tf.device(gpu[0]):
            # Length of sesssions (not considering padding)
            self.seq_len = tf.placeholder(tf.int32, [None], name='seqlen')
            self.batchsize = tf.placeholder(tf.int32, name='batchsize')

            # Average of embeddings session representation
            X_sum = tf.reduce_sum(X_embed, 1)
            self.X_avg = tf.transpose(tf.realdiv(tf.transpose(X_sum), tf.cast(self.seq_len, tf.float32)))

            self.lr = tf.placeholder(tf.float32, name='lr')  # learning rate
            self.pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter

            # Input to inter-session RNN layer
            self.X_lt = tf.placeholder(tf.float32, [None, None, self.LT_INTERNALSIZE],
                                  name='X_lt')  # [BATCHSIZE, LT_INTERNALSIZE]
            self.seq_len_lt = tf.placeholder(tf.int32, [None], name='lt_seqlen')

            # Inter-session RNN
            lt_cell = rnn.GRUCell(self.LT_INTERNALSIZE)
            lt_dropcell = rnn.DropoutWrapper(lt_cell, input_keep_prob=self.pkeep, output_keep_prob=self.pkeep)
            lt_rnn_outputs, lt_rnn_states = tf.nn.dynamic_rnn(lt_dropcell, self.X_lt,
                                                              sequence_length=self.seq_len_lt, dtype=tf.float32)

            # Get the correct outputs (depends on session_lengths)
            last_lt_rnn_output = tf.gather_nd(lt_rnn_outputs, tf.stack([tf.range(self.batchsize), self.seq_len_lt - 1], axis=1))

            # intra-session RNN
            onecell = rnn.GRUCell(self.ST_INTERNALSIZE)
            dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=self.pkeep)
            multicell = rnn.MultiRNNCell([dropcell] * self.N_LAYERS, state_is_tuple=False)
            multicell = rnn.DropoutWrapper(multicell, output_keep_prob=self.pkeep)
            Yr, self.H = tf.nn.dynamic_rnn(multicell, X_embed,
                                      sequence_length=self.seq_len, dtype=tf.float32, initial_state=last_lt_rnn_output)

            self.H = tf.identity(self.H, name='H')  # just to give it a name

            # Apply softmax to the output
            # Flatten the RNN output first, to share weights across the unrolled time steps
            Yflat = tf.reshape(Yr, [-1, self.ST_INTERNALSIZE])  # [ BATCHSIZE x SEQLEN, ST_INTERNALSIZE ]
            # Change from internal size (from RNNCell) to N_ITEMS size
            Ylogits = layers.linear(Yflat, self.N_ITEMS)  # [ BATCHSIZE x SEQLEN, N_ITEMS ]

            # with tf.device(cpu[0]):
            # Flatten expected outputs to match actual outputs
            Y_flat_target = tf.reshape(self.Y_, [-1])  # [ BATCHSIZE x SEQLEN ]

            # Calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Ylogits,
                                                                  labels=Y_flat_target)  # [ BATCHSIZE x SEQLEN ]

            # Mask the losses (so we don't train in padded values)
            mask = tf.sign(tf.to_float(Y_flat_target))
            masked_loss = mask * loss

            # Unflatten loss
            loss = tf.reshape(masked_loss, [self.batchsize, -1])  # [ BATCHSIZE, SEQLEN ]

            # Get the index of the highest scoring prediction through Y
            Y = tf.argmax(Ylogits, 1)  # [ BATCHSIZE x SEQLEN ]
            Y = tf.reshape(Y, [self.batchsize, -1], name='Y')  # [ BATCHSIZE, SEQLEN ]

            # Get prediction
            top_k_values, top_k_predictions = tf.nn.top_k(Ylogits, k=self.TOP_K)  # [BATCHSIZE x SEQLEN, TOP_K]
            self.Y_prediction = tf.reshape(top_k_predictions, [self.batchsize, -1, self.TOP_K], name='YTopKPred')

            # Training
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

            # Stats
            # Average sequence loss
            seqloss = tf.reduce_mean(loss, 1)
            # Average batchloss
            self.batchloss = tf.reduce_mean(seqloss)

        # Average number of correct predictions
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_, tf.cast(Y, tf.int32)), tf.float32))
        loss_summary = tf.summary.scalar("batch_loss", self.batchloss)
        acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
        summaries = tf.summary.merge([loss_summary, acc_summary])

        # Init to save models
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        self.saver = tf.train.Saver(max_to_keep=1)

        # Initialization
        # istate = np.zeros([BATCHSIZE, ST_INTERNALSIZE*N_LAYERS])    # initial zero input state
        self.init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True  # be nice and don't use more memory than necessary
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()


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
        # max_train_length = train_data.groupby([self.session_key])[self.item_key].count().max()
        # max_test_length = test_data.groupby(['SessionId'])['ItemId'].count().max()
        # self.max_length = max(max_train_length, max_test_length)
        self.SEQLEN = train_data.groupby([self.session_key])[self.item_key].count().max()
        self.TOP_K = len(train_data[self.item_key].unique())

        # Load training data
        # home = '..'
        # dataset_path = home + '\\datasets\\lastfm\\4_train_test_split.pickle'
        # date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        # log_file = '.\\testlog\\' + str(date_now) + '-testing-ii-rnn.txt'
        # self.datahandler = IIRNNDataHandler( train_data, test_data, self.user_key, self.item_key, self.session_key, self.time_key, self.BATCHSIZE, log_file, self.MAX_SESSION_REPRESENTATIONS, self.LT_INTERNALSIZE)
        self.datahandler = IIRNNDataHandler( train_data, test_data, self.user_key, self.item_key, self.session_key, self.time_key, self.BATCHSIZE, self.MAX_SESSION_REPRESENTATIONS, self.LT_INTERNALSIZE)
        self.N_ITEMS = self.datahandler.get_num_items() # number of uniq items # warning: original code did not filter items in the test set that did not apeare in the train set! so there are some uniq item which apeared in the test set though they did not apeare in the training set!!! [they both test and training data will be checked, but with this filter we can just consider training data here]
        self.N_SESSIONS = self.datahandler.get_num_training_sessions() # number of sessions in training set
        # itemids = train_data[self.item_key].unique()
        # self.N_ITEMS = len(itemids) # number of uniq items
        # sessions = train_data[self.item_key].unique()
        # self.N_SESSIONS = len(sessions) # number of sessions in training set


        self.init_model()

        ##
        ##  TRAINING
        ##

        print("Starting training.")
        self.sess.run(self.init)
        epoch = 1

        num_training_batches = self.datahandler.get_num_training_batches() # get number of batches in training data with below code

        # num_training_batches = math.ceil(self.N_SESSIONS / self.batch_size)
        # sessions_test = test_data[self.item_key].unique()
        # num_test_batches = math.ceil(len(sessions_test) / self.batch_size)

        while epoch <= self.MAX_EPOCHS:
            print("Starting epoch #" + str(epoch))
            epoch_loss = 0

            self.datahandler.reset_user_batch_data()
            self.datahandler.reset_user_session_representations()
            if self.do_training:
                _batch_number = 0
                xinput, targetvalues, sl, session_reps, sr_sl, user_list = self.datahandler.get_next_train_batch()

                while len(xinput) > int(self.BATCHSIZE / 2):
                    _batch_number += 1
                    batch_start_time = time.time()

                    feed_dict = {self.X: xinput, self.Y_: targetvalues, self.X_lt: session_reps,
                                 self.seq_len_lt: sr_sl, self.lr: self.learning_rate, self.pkeep: self.dropout_pkeep,
                                 self.batchsize: len(xinput), self.seq_len: sl}
                    if self.use_last_hidden_state:
                        _, bl, sess_rep = self.sess.run([self.train_step, self.batchloss, self.H], feed_dict=feed_dict)
                    else:
                        _, bl, sess_rep = self.sess.run([self.train_step, self.batchloss, self.X_avg], feed_dict=feed_dict)

                    self.datahandler.store_user_session_representations(sess_rep, user_list)

                    batch_runtime = time.time() - batch_start_time
                    epoch_loss += bl
                    if _batch_number % 100 == 0:
                        print("Batch number:", str(_batch_number), "/", str(num_training_batches), "| Batch time:",
                              "%.2f" % batch_runtime, " seconds", end='')
                        print(" | Batch loss:", bl, end='')
                        eta = (batch_runtime * (num_training_batches - _batch_number)) / 60
                        eta = "%.2f" % eta
                        print(" | ETA:", eta, "minutes.")

                    xinput, targetvalues, sl, session_reps, sr_sl, user_list = self.datahandler.get_next_train_batch()

                print("Epoch", epoch, "finished")
                print("|- Epoch loss:", epoch_loss)
                epoch += 1

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

        user_id = self.datahandler.get_user_map(input_user_id)

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()
            # in the leave-one-out setting: seems that we do not need the following lines, as we are in the test phase
            if self.user_sess_representation is not None:
                # update the representation for the previous user
                self.datahandler.store_user_session_representations([self.session_representation], [self.user_sess_representation])
            # update the current user id
            self.user_sess_representation = user_id
            # self.datahandler.store_user_session_representations([sess_rep], [user_id])

        # num_test_batches = self.datahandler.get_num_test_batches() # get number of batches in test data with below code
        # xinput, targetvalues, sl, session_reps, sr_sl, user_list = self.datahandler.get_next_test_batch() # TODO: after check values, this line must be deleted
        # xinput: all items of the sessions in the batch
        # targetvalues: all targes items of the sessions in the batch

        # self.datahandler.get_next_batch(self.testset, self.test_session_lengths)

        # feed_dict = {self.X: xinput, self.pkeep: 1.0, self.batchsize: len(xinput), self.seq_len: sl,
        #              self.X_lt: session_reps, self.seq_len_lt: sr_sl}

        # xinput, targetvalues, sl, session_reps, sr_sl, user_list
        # x, y, session_lengths, sess_rep_batch, sess_rep_lengths, user_list
        # session_lengths = self.test_session_lengths[input_user_id][session_id]
        item_id = self.datahandler.get_item_map(input_item_id)
        self.session_items.append(item_id)

        # session_index = self.datahandler.user_next_session_to_retrieve[input_user_id]
        # sl = dataset_session_lengths[input_user_id][session_index]

        # session_lengths = self.datahandler.get_test_session_lengths()
        # sl = session_lengths[user_id][0]

        # sl = session_lengths[user_id][0]
        # sl = [session_lengths]
        sess_rep = self.datahandler.get_user_session_representations(user_id)
        # sess_rep = self.datahandler.user_session_representations[input_user_id]

        # srl = max(self.num_user_session_representations[user], 1)
        # sess_rep_lengths.append(srl)

        # sess_rep_lengths = max(self.datahandler.num_user_session_representations[input_user_id], 1)
        sr_sl = self.datahandler.get_sess_rep_lengths(user_id) #sess_rep_lengths
        # sr_sl =  [sess_rep_lengths]

        # feed_dict = {self.X: [input_item_id], self.pkeep: 1.0, self.batchsize: len([input_item_id]), self.seq_len: [sl],
        #              self.X_lt: [sess_rep], self.seq_len_lt: [sr_sl]}

        # feed_dict = {self.X: [[item_id]], self.pkeep: 1.0, self.batchsize: 1, self.seq_len: [sl],
        #              self.X_lt: [sess_rep], self.seq_len_lt: [sr_sl]}

        feed_dict = {self.X: [self.session_items], self.pkeep: 1.0, self.batchsize: 1, self.seq_len: [len(self.session_items)],
                     self.X_lt: [sess_rep], self.seq_len_lt: [sr_sl]}

        if self.use_last_hidden_state:
            batch_predictions, sess_rep = self.sess.run([self.Y_prediction, self.H], feed_dict=feed_dict)
        else:
            batch_predictions, sess_rep = self.sess.run([self.Y_prediction, self.X_avg], feed_dict=feed_dict)

        self.session_representation = sess_rep

        # pred_series = pd.Series()
        # for i in range(predictions.size):
        #     value = (self.TOP_K - i)*0.1
        #     pred_series.add(value)

        # batch_predictions[0][0] is ndarray
        # predictions = batch_predictions[0][0]
        predictions = batch_predictions[0][len(self.session_items)-1] # just consider the predictions for the last item

        # map reverse item ids
        # convert ndarray to Series [item ids : index, values: scores]
        pred_list = []
        score_list = []
        k = self.TOP_K
        for pred in predictions:
            pred_list.append(self.datahandler.get_item_map_reverse(pred))
            score = k
            score_list.append(score)
            k = k - 1

        pred_series = pd.Series(score_list, index=pred_list)
        # predictions = self.datahandler.get_item_map_reverse(predictions)
        # pred_series = pd.Series(pred_list, index=pred_list)
        # k = self.TOP_K
        # for index, value in pred_series.items():
        #     score = k*0.1
        #     pred_series.replace(value, score, inplace=True)
        #     k = k-1

        # return batch_predictions
        return pred_series
        # numpy_data = np.array([[1, 2], [3, 4]])
        # df = pd.DataFrame(data=numpy_data, index=["row1", "row2"], columns=["column1", "column2"])

    #SR-GNN
    # scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
    # self.predicted_item_ids = []
    # scores = scores[0]
    # for idx in range(len(scores)):
    #     self.predicted_item_ids.append(
    #         int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)
    #
    #
    # series = pd.Series(data=scores, index=self.predicted_item_ids)
    # return series


    def clear(self):
        self.sess.close()
        tf.reset_default_graph()
        pass


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