import sys
sys.path.append('../..')  # noqa


import tensorflow as tf
from tensorflow.contrib.rnn import *


class UserGruModel():
    def __init__(self, config):
        self.config = config

        # Data stats
        self._num_users = config['num_users']
        self._num_items = config['num_items']
        self._max_length = config['max_length']

        # Hyper parameters
        self._cell = config['cell']
        self._entity_embedding = config['entity_embedding']
        self._context_embedding = config['context_embedding']
        self._hidden_units = config['hidden_units']
        self._num_layers = config['num_layers']

        # Input
        self._combination = config['combination']
        self._fusion_type = config['fusion_type']

        # Placeholder
        self.user = tf.placeholder(tf.int32, shape=[None, self._max_length])
        self.item = tf.placeholder(tf.int32, shape=[None, self._max_length])
        self.day_of_week = tf.placeholder(
            tf.int32, shape=[None, self._max_length])
        self.month_period = tf.placeholder(
            tf.int32, shape=[None, self._max_length])
        self.next_items = tf.placeholder(
            tf.int32, shape=[None, self._max_length])
        self.labels = tf.one_hot(depth=self._num_items + 1,
                                 indices=self.next_items, dtype=tf.int32)
        self.keep_pr = tf.placeholder(tf.float32)

        self.length = tf.reduce_sum(tf.sign(self.next_items), axis=1)
        self.global_step = tf.Variable(0, name="global_step",
                                       trainable=False)

        # Model variable
        self._E = {}
        self._embs = {}
        self._rnn_cell = None
        self._w = {}
        self._b = {}
        self._Va = {}
        self._ba = {}
        self._alpha = []

        # Output
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self._logits = None
        self._output_prob = None

        self.build_model()
        self.print_info()

    def print_info(self):
        print('--- Model info ---')
        print('- Model name: ', self.config['name'])
        print('- Num users: ', self._num_users)
        print('- Num items: ', self._num_items)
        print('- Input type: ', self._combination)
        print('- Fusion type: ', self._fusion_type)
        print('- Max session length: ', self._max_length)
        print('- Entity embedding: ', self._entity_embedding)
        print('- Time embedding: ', self._context_embedding)
        print('- Hidden unit: ', self._hidden_units)
        print('- Num layers: ', self._num_layers)
        print('- RNN cell: ', self._cell)

    def build_model(self):
        with tf.variable_scope('embeddings'):
            for x, y, k in zip([self._num_items + 1,
                                self._num_users + 1, 8, 25],
                               [self._entity_embedding] * 2 +
                               [self._context_embedding] * 2,
                               ['i', 'u', 'd', 'm']):
                self._E[k] = tf.get_variable(shape=[x, y],
                                             name='E' + k, dtype=tf.float32)
        for v, k in zip([self.item, self.user,
                         self.day_of_week, self.month_period],
                        ['i', 'u', 'd', 'm']):
            self._embs[k] = tf.nn.embedding_lookup(self._E[k], v)

        self._embs['u'] = tf.nn.dropout(self._embs['u'], self.keep_pr)
        self._embs['i'] = tf.nn.dropout(self._embs['i'], self.keep_pr)

        with tf.variable_scope('rnn-cell'):
            if self._cell == 'gru':
                self._rnn_cell = MultiRNNCell(
                    [GRUCell(self._hidden_units)
                        for _ in range(self._num_layers)])
            elif self._cell == 'lstm':
                self._rnn_cell = MultiRNNCell(
                    [LSTMCell(self._hidden_units)
                        for _ in range(self._num_layers)])
            else:
                self._rnn_cell = MultiRNNCell(
                    [RNNCell(self._hidden_units)
                        for _ in range(self._num_layers)])

        if self._fusion_type == 'pre':
            self._logits = self._pre_fusion()
        elif self._fusion_type == 'post':
            self._logits = self._post_fusion()
        else:
            print('Unkown fusion type')
            exit()

        self._output_prob = tf.nn.softmax(self._logits)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self._logits)
        self.loss = tf.reduce_mean(self.loss)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config['learning_rate'])
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step)

    def clear(self):
        # Model variable
        self._E = {}
        self._embs = {}
        self._rnn_cell = None
        self._w = {}
        self._b = {}
        self._Va = {}
        self._ba = {}
        self._alpha = []

        # Output
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self._logits = None
        self._output_prob = None


    def _feed_forward(self, inputs, output_size, key, activation=None):
        with tf.name_scope('feedforward_' + key):
            if key in self._w.keys():
                print('Variable with key w_%s already exists' % key)
                exit(0)
            self._w[key] = tf.get_variable(
                shape=[int(inputs.shape[-1]), output_size],
                name='w_' + key, dtype=tf.float32)
            self._b[key] = tf.get_variable(
                shape=[output_size], name='b_' + key, dtype=tf.float32)
            output = tf.nn.xw_plus_b(inputs, self._w[key], self._b[key])
            if activation is None:
                return output
            return activation(output)

    def _pre_fusion(self):
        if self._combination == 'linear':
            inputs = tf.concat([self._embs['i'], self._embs['u']], 2)
        elif self._combination == 'linear-context':
            inputs = tf.concat([self._embs['i'], self._embs['u'],
                                self._embs['d'], self._embs['m']], 2)
        elif self._combination == 'adaptive':
            inputs = self._adaptive_gate(self._embs['i'], self._embs['u'])
        elif self._combination == 'adaptive-context':
            inputs = self._adaptive_gate_context(
                self._embs['i'], self._embs['u'], self._embs['d'],
                self._embs['m'])
        elif self._combination == 'weighted':
            inputs = self._weighted_gate(self._embs['i'], self._embs['u'])
        else:
            print('Unrecognize input type.Exit')
            exit(0)

        output_states, _ = tf.nn.dynamic_rnn(
            self._rnn_cell, inputs, sequence_length=self.length,
            dtype=tf.float32)
        output_states = tf.reshape(output_states, [-1, self._hidden_units])

        self._logits = self._feed_forward(
            output_states, self._num_items + 1, key='fc')

        return self._logits

    def _post_fusion(self):
        output_states, _ = tf.nn.dynamic_rnn(
            self._rnn_cell, self._embs['i'], sequence_length=self.length,
            dtype=tf.float32)
        if self._combination == 'linear':
            final_state = tf.reshape(
                tf.concat([output_states, self._embs['u']], -1),
                [-1, self._hidden_units + self._entity_embedding])
        elif self._combination == 'linear-context':
            final_state = tf.reshape(
                tf.concat([output_states, self._embs['u'],
                           self._embs['d'], self._embs['m']], -1),
                [-1, self._hidden_units + self._entity_embedding +
                 2 * self._context_embedding])
        elif self._combination == 'adaptive':
            final_state = self._adaptive_gate(output_states, self._embs['u'])
            final_state = tf.reshape(
                final_state, [-1, self._hidden_units + self._entity_embedding])
        elif self._combination == 'adaptive-context':
            final_state = self._adaptive_gate_context(
                output_states, self._embs['u'], self._embs['d'],
                self._embs['m'])
            final_state = tf.reshape(
                final_state,
                [-1, self._hidden_units + self._entity_embedding +
                 2 * self._context_embedding])
        elif self._combination == 'weighted':
            final_state = self._weighted_gate(output_states, self._embs['u'])
            final_state = tf.reshape(
                final_state,
                [-1, self._hidden_units + self._entity_embedding])
        elif self._combination == 'voting':
            output_states = tf.reshape(output_states, [-1, self._hidden_units])
            user_embs = tf.reshape(self._embs['u'],
                                   [-1, self._entity_embedding])
            vote_user = self._feed_forward(
                user_embs, self._num_items + 1, key='vote_u',
                activation=tf.nn.softmax)
            vote_item = self._feed_forward(
                output_states, self._num_items + 1, key='vote_i',
                activation=tf.nn.softmax)
            self._logits = vote_user + vote_item
            return self._logits
        else:
            print('Unrecognize input type.Exit')
            exit(0)

        self._logits = self._feed_forward(
            final_state, self._num_items + 1, key='fc')

        return self._logits

    def _adaptive_gate_context(self, item, user, day, month):
        with tf.name_scope('adaptive'):
            for x, k in zip([self._hidden_units, self._entity_embedding] +
                            [self._context_embedding] * 2,
                            ['i', 'u', 'd', 'm']):
                self._Va[k] = tf.get_variable(shape=[x],
                                              name='Va_' + k, dtype=tf.float32)

            for k in ['i', 'u', 'd', 'm']:
                self._ba[k] = tf.get_variable(shape=[], name='ba_' + k,
                                              dtype=tf.float32)

        alpha = []
        for x, k in zip([item, user, day, month],
                        ['i', 'u', 'd', 'm']):
            alpha.append(tf.sigmoid(tf.reduce_sum(
                tf.cast(x, tf.float32) * self._Va[k], axis=2) + self._ba[k]))

        self._alpha = []
        for t in range(self._max_length):
            wt = []
            for i in range(4):
                wt.append(alpha[i][:, t])
            sum_exp = tf.reduce_sum(tf.exp(wt), axis=0)
            self._alpha.append([tf.exp(w_) / sum_exp for w_ in wt])

        self._alpha = tf.transpose(tf.stack(self._alpha), [2, 0, 1])
        final_input = []
        for i, x in enumerate([item, user, day, month]):
            final_input.append(tf.expand_dims(self._alpha[:, :, i], dim=2) * x)
        return tf.concat(final_input, -1)

    def _adaptive_gate(self, item, user):
        with tf.name_scope('adaptive_gate'):
            item = tf.reshape(item, shape=[-1, item.shape[-1]])
            item = self._feed_forward(item, self._hidden_units, key='a_item',
                                      activation=tf.nn.tanh)
            item = tf.reshape(item, shape=[-1, self._max_length,
                                           item.shape[-1]])

            user = tf.reshape(user, shape=[-1, user.shape[-1]])
            user = self._feed_forward(user, self._hidden_units, key='a_user',
                                      activation=tf.nn.tanh)
            user = tf.reshape(user, shape=[-1, self._max_length,
                                           user.shape[-1]])
            for x, k in zip([self._hidden_units, self._hidden_units],
                            ['i', 'u']):
                self._Va[k] = tf.get_variable(shape=[x],
                                              name='Va_' + k, dtype=tf.float32)

            for k in ['i', 'u']:
                self._ba[k] = tf.get_variable(shape=[], name='ba_' + k,
                                              dtype=tf.float32)

        alpha = []
        for x, k in zip([item, user], ['i', 'u']):
            alpha.append(tf.reduce_sum(
                tf.cast(x, tf.float32) * self._Va[k], axis=2) + self._ba[k])

        self._alpha = []
        for t in range(self._max_length):
            wt = []
            for i in range(2):
                wt.append(alpha[i][:, t])
            sum_exp = tf.reduce_sum(tf.exp(wt), axis=0)
            self._alpha.append([tf.exp(w_) / sum_exp for w_ in wt])
        self._alpha = tf.transpose(tf.stack(self._alpha), [2, 0, 1])
        final_input = []
        for i, x in enumerate([item, user]):
            final_input.append(tf.expand_dims(self._alpha[:, :, i], dim=2) * x)

        return tf.concat(final_input, -1)

    def _weighted_gate(self, item, user):
        with tf.name_scope('weighted_gate'):
            for k in ['i', 'u']:
                self._Va[k] = tf.get_variable(shape=[1],
                                              name='Va_' + k, dtype=tf.float32)
            attention_w = tf.nn.softmax(list(self._Va.values()))
        item = item * attention_w[0]
        user = user * attention_w[1]
        return tf.concat([item, user], -1)

    def get_training_vars(self):
        return self.train_op, self.loss, self.global_step

    def get_output(self):
        return self._output_prob

    def get_attention_weight(self):
        return self._alpha
