# coding=utf-8

from __future__ import print_function
import numpy as np
import tensorflow as tf
from time import time
from algorithms.CSRM.ome import OME

import time
import pandas as pd
import pickle
tf.set_random_seed(42)
np.random.seed(42)

def numpy_floatX(data):
    return np.asarray(data, dtype=np.float32)

class CSRM:
    def __init__(self,
                 dim_proj=100,
                 hidden_units=100,
                 patience=5,
                 memory_size=512,
                 memory_dim=100,
                 shift_range=1,
                 controller_layer_numbers=0,
                 batch_size=512,
                 epoch=15,
                 lr=0.0005,
                 keep_probability='[0.75,0.5]',
                 no_dropout='[1.0,1.0]',
                 display_frequency=200,
                 session_key='SessionId', item_key='ItemId'
                 ):

        self.session = -1
        self.session_key = session_key
        self.item_key = item_key
        self.dim_proj = dim_proj
        self.hidden_units = hidden_units
        self.patience = patience
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.shift_range = shift_range
        self.controller_layer_numbers = controller_layer_numbers
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.keep_probability = np.array([0.75, 0.5])
        self.no_dropout = np.array([1.0, 1.0])
        self.display_frequency = display_frequency
        self.controller_hidden_layer_size = 100
        self.controller_output_size = self.memory_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1 + self.memory_dim * 3 + 1 + 1 + (self.shift_range * 2 + 1) + 1






    def build_graph(self):
        self.params = self.init_params()

        self.x_input = tf.placeholder(tf.int64, [None, None])
        self.mask_x = tf.placeholder(tf.float32, [None, None])
        self.y_target = tf.placeholder(tf.int64, [None])
        self.len_x = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32, [None])
        self.starting = tf.placeholder(tf.bool)

        """       
        attention gru & global gru 实现
        Output:
        global_session_representation
        attentive_session_represention
        """
        self.n_timesteps = tf.shape(self.x_input)[1]
        self.n_samples = tf.shape(self.x_input)[0]

        emb = tf.nn.embedding_lookup(self.params['Wemb'], self.x_input)
        emb = tf.nn.dropout(emb, keep_prob=self.keep_prob[0])

        with tf.variable_scope('global_encoder'):
            cell_global = tf.nn.rnn_cell.GRUCell(self.hidden_units)
            init_state = cell_global.zero_state(self.n_samples, tf.float32)
            outputs_global, state_global = tf.nn.dynamic_rnn(cell_global, inputs=emb, sequence_length=self.len_x,
                                                             initial_state=init_state, dtype=tf.float32)
            last_global = state_global  # batch_size*hidden_units

        with tf.variable_scope('local_encoder'):
            cell_local = tf.nn.rnn_cell.GRUCell(self.hidden_units)
            init_statel = cell_local.zero_state(self.n_samples, tf.float32)
            outputs_local, state_local = tf.nn.dynamic_rnn(cell_local, inputs=emb, sequence_length=self.len_x,
                                                           initial_state=init_statel, dtype=tf.float32)
            last_h = state_local  # batch_size*hidden_units

            tmp_0 = tf.reshape(outputs_local, [-1, self.hidden_units])
            tmp_1 = tf.reshape(tf.matmul(tmp_0, self.params['W_encoder']),
                               [self.n_samples, self.n_timesteps, self.hidden_units])
            tmp_2 = tf.expand_dims(tf.matmul(last_h, self.params['W_decoder']), 1)  # batch_size*hidden_units
            tmp_3 = tf.reshape(tf.sigmoid(tmp_1 + tmp_2), [-1, self.hidden_units])  # batch_size,n_steps, hidden_units
            alpha = tf.matmul(tmp_3, tf.transpose(self.params['bl_vector']))
            res = tf.reduce_sum(alpha, axis=1)
            sim_matrix = tf.reshape(res, [self.n_samples, self.n_timesteps])

            att = tf.nn.softmax(sim_matrix * self.mask_x) * self.mask_x  # batch_size*n_step
            p = tf.expand_dims(tf.reduce_sum(att, axis=1), 1)
            weight = att / p
            atttention_proj = tf.reduce_sum((outputs_local * tf.expand_dims(weight, 2)), 1)
            
        self.global_session_representation = last_global
        self.attentive_session_represention = atttention_proj

        # 初始化ntm_cell，用于读写memory
        self.ome_cell = OME(mem_size=(self.memory_size, self.memory_dim), shift_range=self.shift_range,
                            hidden_units=self.hidden_units)

        # 创建用于存放读写memory的state的placeholder
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_units])
        self.memory_network_reads, self.memory_new_state = self.ome_cell(self.state, atttention_proj, self.starting)

        att_mean, att_var = tf.nn.moments(self.attentive_session_represention, axes=[1])
        self.attentive_session_represention = (self.attentive_session_represention - tf.expand_dims(att_mean, 1)) / tf.expand_dims(tf.sqrt(att_var + 1e-10), 1)
        glo_mean, glo_var = tf.nn.moments(self.global_session_representation, axes=[1])
        self.global_session_representation = (self.global_session_representation - tf.expand_dims(glo_mean, 1)) / tf.expand_dims(tf.sqrt(glo_var + 1e-10), 1)
        ntm_mean, ntm_var = tf.nn.moments(self.memory_network_reads, axes=[1])
        self.memory_network_reads = (self.memory_network_reads - tf.expand_dims(ntm_mean, 1)) / tf.expand_dims(tf.sqrt(ntm_var + 1e-10), 1)

        new_gate = tf.matmul(self.attentive_session_represention, self.params['inner_encoder']) + \
                   tf.matmul(self.memory_network_reads, self.params['outer_encoder']) + \
                   tf.matmul(self.global_session_representation, self.params['state_encoder'])
        new_gate = tf.nn.sigmoid(new_gate)
        self.narm_representation = tf.concat((self.attentive_session_represention, self.global_session_representation), axis=1)
        self.memory_representation = tf.concat((self.memory_network_reads, self.memory_network_reads), axis=1)
        final_representation = new_gate * self.narm_representation + (1 - new_gate) * self.memory_representation

        # prediction
        proj = tf.nn.dropout(final_representation, keep_prob=self.keep_prob[1])
        ytem = tf.matmul(self.params['Wemb'], self.params['bili'])   # [n_items, 200]
        hypothesis = tf.matmul(proj, tf.transpose(ytem)) + 1e-10 # [batch_size, n_step, n_items]
        self.hypo = tf.nn.softmax(hypothesis)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.y_target))
        # optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=1)

    def init_weights(self, i_name, shape):
        sigma = np.sqrt(2. / shape[0])
        return tf.get_variable(name=i_name, dtype=tf.float32, initializer=tf.random_normal(shape) * sigma)

    def init_params(self):
        """
        Global (not GRU) parameter. For the embeding and the classifier.
        """
        params = dict()
        # embedding
        params['Wemb'] = self.init_weights('Wemb', (self.n_items, self.dim_proj))
        # attention
        params['W_encoder'] = self.init_weights('W_encoder', (self.hidden_units, self.hidden_units))
        params['W_decoder'] = self.init_weights('W_decoder', (self.hidden_units, self.hidden_units))
        params['bl_vector'] = self.init_weights('bl_vector', (1, self.hidden_units))
        # classifier
        params['bili'] = self.init_weights('bili', (self.dim_proj, 2 * self.hidden_units))
        # final gate
        params['inner_encoder'] = self.init_weights('inner_encoder', (self.hidden_units, 1))
        params['outer_encoder'] = self.init_weights('outer_encoder', (self.hidden_units, 1))
        params['state_encoder'] = self.init_weights('state_encoder', (self.hidden_units, 1))

        return params

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:  minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if minibatch_start != n:
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def create_training_data(self, data, test):
        
        data['maxtime'] = data.groupby(self.session_key).Time.transform(max)
        test['maxtime'] = test.groupby(self.session_key).Time.transform(max)
        data.sort_values( ['maxtime',self.session_key,'Time'], inplace=True )
        test.sort_values( ['maxtime',self.session_key,'Time'], inplace=True )
        #data.sort_values( [self.session_key,'Time'], inplace=True )
        #test.sort_values( [self.session_key,'Time'], inplace=True )
        
        del data['maxtime'], test['maxtime']
        
        index_session = data.columns.get_loc(self.session_key)
        index_item = data.columns.get_loc('ItemIdx')
        
        out_seqs_tr = []
        labs_tr = []

        session = -1
        session_items = []

        for row in data.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                session = row[index_session]
                session_items = list()

            session_items.append(row[index_item])

            if len(session_items) > 1:
                out_seqs_tr += [session_items[:-1]]
                labs_tr += [session_items[-1]]
        
        
        index_session = test.columns.get_loc(self.session_key)
        index_item = test.columns.get_loc('ItemIdx')
        
        out_seqs_te = []
        labs_te = []

        session = -1
        session_items = []

        for row in test.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                session = row[index_session]
                session_items = list()

            session_items.append(row[index_item])

            if len(session_items) > 1:
                out_seqs_te += [session_items[:-1]]
                labs_te += [session_items[-1]]
        
        return (out_seqs_tr, labs_tr), (out_seqs_te, labs_te)

    def prepare_data(self,seqs, labels):
        np.random.seed(42)
        """Create the matrices from the datasets.

        This pad each sequence to the same lenght: the lenght of the
        longuest sequence or maxlen.

        if maxlen is set, we will cut all sequence to this maximum
        lenght.

        This swap the axis!
        """
        # x: a list of sentences

        lengths = [len(s) for s in seqs]
        n_samples = len(seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen), dtype=np.int64)
        x_mask = np.ones((n_samples, maxlen), dtype=np.float32)
        for idx, s in enumerate(seqs):
            x[idx, :lengths[idx]] = s

        x_mask *= (1 - (x == 0))  # 将x的非0元素变为1
        # seq_length = [i if i <= maxlen else maxlen for i in lengths]

        return x, x_mask, labels, lengths

    def load_data(self, train_set, valid_portion=0.1, maxlen=False, sort_by_len=False):
        '''Loads the dataset
        :type path: String
        :param path: The path to the dataset (here RSC2015)
        :type n_items: int
        :param n_items: The number of items.
        :type valid_portion: float
        :param valid_portion: The proportion of the full train set used for
            the validation set.
        :type maxlen: None or positive int
        :param maxlen: the max sequence length we use in the train/valid set.
        :type sort_by_len: bool
        :name sort_by_len: Sort by the sequence lenght for the train,
            valid and test set. This allow faster execution as it cause
            less padding per minibatch. Another mechanism must be used to
            shuffle the train set at each epoch.
        '''

        #############
        # LOAD DATA #
        #############

        if maxlen:
            new_train_set_x = []
            new_train_set_y = []
            for x, y in zip(train_set[0], train_set[1]):
                if len(x) < maxlen:
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
                else:
                    new_train_set_x.append(x[:maxlen])
                    new_train_set_y.append(y)
            train_set = (new_train_set_x, new_train_set_y)
            del new_train_set_x, new_train_set_y

        # split training set into validation set
        train_set_x, train_set_y = train_set
        n_samples = len(train_set_x)
        sidx = np.arange(n_samples, dtype='int32')
        #np.random.shuffle(sidx)
        n_train = int(np.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)

        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort_by_len:
            sorted_index = len_argsort(valid_set_x)
            valid_set_x = [valid_set_x[i] for i in sorted_index]
            valid_set_y = [valid_set_y[i] for i in sorted_index]

        train = (train_set_x, train_set_y)
        valid = (valid_set_x, valid_set_y)

        return train, valid
    
    def load_test(self, test_set, maxlen=None):
        '''Loads the dataset
        :type path: String
        :param path: The path to the dataset (here RSC2015)
        :type n_items: int
        :param n_items: The number of items.
        :type valid_portion: float
        :param valid_portion: The proportion of the full train set used for
            the validation set.
        :type maxlen: None or positive int
        :param maxlen: the max sequence length we use in the train/valid set.
        :type sort_by_len: bool
        :name sort_by_len: Sort by the sequence lenght for the train,
            valid and test set. This allow faster execution as it cause
            less padding per minibatch. Another mechanism must be used to
            shuffle the train set at each epoch.
        '''

        #############
        # LOAD DATA #
        #############

        if maxlen:
            new_test_set_x = []
            new_test_set_y = []
            for x, y in zip(test_set[0], test_set[1]):
                if len(x) < maxlen:
                    new_test_set_x.append(x)
                    new_test_set_y.append(y)
                else:
                    new_test_set_x.append(x[:maxlen])
                    new_test_set_y.append(y)
            test_set = (new_test_set_x, new_test_set_y)
            del new_test_set_x, new_test_set_y

        return test_set
    
    def construct_feeddict(self, batch_data, batch_label, keepprob, state, starting=False):
        x, mask, y, lengths = self.prepare_data(batch_data, batch_label)
        feed = {self.x_input: x, self.mask_x: mask, self.y_target: y, self.len_x: lengths, self.keep_prob: keepprob,
                self.state: state, self.starting: starting}
        # feed the initialized state into placeholder

        return feed


    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        
        with tf.Graph().as_default():
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            
            self.n_items = len(data[self.item_key].unique()) + 1
            self.build_graph()
    
            nis = data[self.item_key].nunique()
    
            self.itemmap = pd.Series(index=data[self.item_key].unique(), data=range(1, nis + 1))
            data = data.merge(self.itemmap.to_frame('ItemIdx'), how='inner', right_index=True, left_on=self.item_key)
            #data.sort_values(['SessionId', 'Time'], inplace=True)
            
            test = test.merge(self.itemmap.to_frame('ItemIdx'), how='inner', right_index=True, left_on=self.item_key)
            #test.sort_values(['SessionId', 'Time'], inplace=True)
            
            self.traindata, self.testdata = self.create_training_data(data, test)
            self.dataload = (self.load_data, self.load_test)
            #self.layers = {'gru': (self.param_init_gru, self.gru_layer)}
    
            self.train_gru()

    #def fit(self, Train_data, Validation_data, Test_data, result_path='save/'):
    def train_gru(self):
                
        self.train_loss_record = []
        self.valid_loss_record = []
        self.test_loss_record = []

        self.train_recall_record, self.train_mrr_record = [], []
        self.valid_recall_record, self.valid_mrr_record = [], []
        self.test_recall_record, self.test_mrr_record = [], []
        
        load_data, load_test = self.get_dataset()

        train, valid = load_data(self.traindata)
        test = load_test(self.testdata)

        # 初始化参数
        print(" [*] Initialize all variables")
        self.sess.run(tf.global_variables_initializer())
        print(" [*] Initialization finished")

        uidx = 0
        bad_count = 0
        estop = False
        for epoch in range(self.epoch):
            kf = self.get_minibatches_idx(len(train[0]), self.batch_size)
            kf_valid = self.get_minibatches_idx(len(valid[0]), self.batch_size)
            kf_test = self.get_minibatches_idx(len(test[0]), self.batch_size)

            start_time = time.time()
            nsamples = 0
            epoch_loss = []
            session_memory_state = np.random.normal(0, 0.05, size=[1, self.hidden_units])
            starting = True
            # 训练
            print('*****************************************************************')
            for _, train_index in kf:
                uidx += 1
                # Select the random examples for this minibatch
                batch_label = [train[1][t] for t in train_index]
                batch_data = [train[0][t] for t in train_index]
                nsamples += len(batch_label)
                feed_dict = self.construct_feeddict(batch_data, batch_label, self.keep_probability, session_memory_state, starting)
                cost, _, session_memory_state = self.sess.run([self.loss, self.optimizer, self.memory_new_state], feed_dict=feed_dict)
                starting = False

                epoch_loss.append(cost)
                if np.mod(uidx, self.display_frequency) == 0:
                    print('Epoch ', epoch, 'Update ', uidx, 'Loss ', np.mean(epoch_loss))

            valid_evaluation, session_memory_state = self.pred_evaluation(valid, kf_valid, session_memory_state)
            self.valid_recall_record.append(valid_evaluation[0])
            self.valid_mrr_record.append(valid_evaluation[1])

            if valid_evaluation[0] >= np.array(self.valid_recall_record).max():
                bad_count = 0
                print('Best perfomance updated!')
                #self.saver.save(self.sess, result_path + "/model.ckpt")
                #pickle.dump(session_memory_state, open('save/lastfm_memory.pkl', 'w'))

            test_evaluation, session_memory_state = self.pred_evaluation(test, kf_test, session_memory_state)
            self.test_recall_record.append(test_evaluation[0])
            self.test_mrr_record.append(test_evaluation[1])
            print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1], 
                  '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])

            if valid_evaluation[0] < np.array(self.valid_recall_record).max():
                bad_count += 1
                print('===========================>Bad counter: ' + str(bad_count))
                print('current validation recall: ' + str(valid_evaluation[0]) +
                      '      history max recall:' + str(np.array(self.valid_recall_record).max()))
                if bad_count >= self.patience:
                    print('Early Stop!')
                    estop = True
            end_time = time.time()
            print('Seen %d samples' % nsamples)
            print(('This epoch took %.1fs' % (end_time - start_time)))
            print('*****************************************************************')
            if estop:
                break

        p = self.valid_recall_record.index(np.array(self.valid_recall_record).max())
        print('=================Best performance=================')
        print('Valid Recall@20:', self.valid_recall_record[p], '   Valid Mrr@20:', self.valid_mrr_record[p], 
             '\nTest Recall@20', self.test_recall_record[p], '   Test Mrr@20:', self.test_mrr_record[p])
        print('==================================================')
        
        self.last_state = session_memory_state

    def get_dataset(self):
        return self.dataload[0], self.dataload[1]

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False,
                     type='view'):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        if (self.session != session_id):  # new session

            self.session = session_id
            self.session_items = list()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return


        x = [self.itemmap[self.session_items].values]
        y = self.itemmap[self.session_items].values.tolist()

        #x, mask, y = self.prepare_data(x, y)
        preds, session_memory_state = self.pred_function( x, y, ntm_init_state=self.last_state )
        self.last_state = session_memory_state

        return pd.Series(data=preds[0][1:], index=self.itemmap.index)

    def pred_function(self, seqs, label, ntm_init_state=None):
        start = False
        if ntm_init_state is None:
            start = True
            ntm_init_state = np.random.normal(0, 0.05, size=[1, self.hidden_units])
        #ntm_init_state = np.random.normal(0, 0.05, size=[1, len(seqs)])
        batch_data = seqs
        batch_label = label
        feed_dict = self.construct_feeddict(batch_data, batch_label, self.no_dropout, ntm_init_state, start)
        preds, ntm_init_state = self.sess.run([self.hypo, self.memory_new_state], feed_dict=feed_dict)
                
        return preds, ntm_init_state

    def pred_evaluation(self, data, iterator, ntm_init_state):
        """
        Compute recall@20 and mrr@20
        f_pred_prob: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        recall = 0.0
        mrr = 0.0
        evalutation_point_count = 0

        for _, valid_index in iterator:
            batch_data = [data[0][t] for t in valid_index]
            batch_label = [data[1][t] for t in valid_index]
            feed_dict = self.construct_feeddict(batch_data, batch_label, self.no_dropout, ntm_init_state)
            pred, ntm_init_state = self.sess.run([self.hypo, self.memory_new_state], feed_dict=feed_dict)
            ranks = (pred.T > np.diag(pred.T[batch_label])).sum(
                axis=0) + 1  # np.diag(preds.T[targets]) each bacth target"s score
            rank_ok = (ranks <= 20)  # tf.diag_part(input, name=None)
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)

        recall = numpy_floatX(recall) / evalutation_point_count
        mrr = numpy_floatX(mrr) / evalutation_point_count
        eval_score = (recall, mrr)

        return eval_score, ntm_init_state


    def clear(self):
        self.sess.close()
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
        return False

