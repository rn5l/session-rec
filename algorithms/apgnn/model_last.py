
import tensorflow as tf
import math
import random
import numpy as np
import pandas as pd
from .transformer import encoder, decoder, multihead_attention,normalize


class Model(object):
    def __init__(self, hidden_size=100, user_size=100, batch_size=100, seq_max=20, group_max=100, mode='usual_attention', data=None, decoder_attention=True,
                 encoder_attention=True, user_=True, behaviour_=False,
                 history_=True, sparse=True):
        self.hidden_size = hidden_size
        self.user_size = user_size
        self.batch_size = batch_size
        self.seq_max = seq_max
        self.group_max = group_max
        self.mode = mode
        self.data = data
        self.decoder_attention = decoder_attention
        self.encoder_attention = encoder_attention
        self.history_ = history_
        self.user_ = user_
        self.behaviour_ = behaviour_
        self.sparse = sparse
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        if self.mode == 'transformer':
            self.control_dim = 1
        if self.mode == 'usual_attention' or 'transformer' or 'attention':
            self.u_w1 = tf.get_variable('u_w1', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.u_w2 = tf.get_variable('u_w2', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.u_v = tf.get_variable('u_v', [1, self.hidden_size], dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.u_b = tf.get_variable('u_b', [self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.control_dim = 2
        if self.history_:
            self.h_w1 = tf.get_variable('h_w1', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.h_w2 = tf.get_variable('h_w2', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.h_v = tf.get_variable('h_v', [1, self.hidden_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.h_b = tf.get_variable('h_b', [self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.control_dim += 2

    def session_embed(self, embed, session_alias, session_mask, session_len, user, train=True):
        session = tf.stack([tf.nn.embedding_lookup(embed[i], session_alias[i]) for i in range(self.batch_size)], 0)
        if self.pool == 'mean':
            session_seq = tf.div(tf.reduce_sum(session, axis=2), tf.to_float(tf.expand_dims(session_mask, 2)))
        elif self.pool == 'max':
            session_seq = tf.reduce_max(session, axis=2)
        session_embed_mask = tf.sequence_mask(session_len, maxlen=self.group_max, dtype=tf.float32)
        if self.encoder_attention:
            session_seq = multihead_attention(tf.reshape(session_seq, [self.batch_size, -1, self.hidden_size]),
                                              session_embed_mask,
                                              tf.reshape(session_seq, [self.batch_size, -1, self.hidden_size]),
                                              session_embed_mask, self.hidden_size, is_training=train)
        # elif self.mode == 'transformer':
        #     session_seq = encoder(tf.reshape(session_seq, [self.batch_size, -1, self.hidden_size]),
        #                           session_embed_mask, self.group_max, self.hidden_size, train=train)
        # --------------stamp attention------------------
        session_last = tf.gather_nd(session_seq,
                                    tf.stack([tf.range(self.batch_size, dtype=tf.int64), session_len - 1], axis=1))
        session_ma = stamp_attention(session_seq, session_last, session_embed_mask, self.h_w1, self.h_w2, self.h_b,
                                     self.h_v, self.hidden_size, self.batch_size )
        if self.mode == 'transformer' or self.mode == 'attention':
            return tf.reshape(session_seq, [self.batch_size, -1, self.hidden_size]), session_ma
        else:
            return session_ma

    def forward(self, adj_in, adj_out, items, seq_alias, seq_mask,
                session_alias, session_len, session_mask, tar, user, train=True):
        if self.graph == 'ggnn':
            re_embedding = self.ggnn(items, user, adj_in, adj_out, is_training=train)
        elif self.graph == 'no_graph':
            re_embedding = self.no_graph(items)
        b = self.embedding[1:]
        with tf.variable_scope('forward'):
            if self.mode == 'transformer':
                with tf.variable_scope('transformer'):
                    session_embed_mask = tf.sequence_mask(session_len, maxlen=self.group_max, dtype=tf.float32)
                    encoder_out, session_h = self.session_embed(
                        re_embedding, session_alias, session_mask, session_len, user, train=train)
                    decoder_input = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], 0)
                    dec_mask = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    decoder_out = mul_attention(
                        tf.reshape(decoder_input, [self.batch_size, -1, self.hidden_size]),
                        dec_mask,
                        tf.reshape(encoder_out, [self.batch_size, -1, self.hidden_size]),
                        session_embed_mask,
                        self.hidden_size, data=self.data)
                    #stamp
                    decoder_last = tf.gather_nd(decoder_out,
                                                tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask-1], axis=1))
                    ma = stamp_attention(decoder_out, decoder_last,  dec_mask, self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        ma = tf.concat([ma, session_h], -1)
            elif self.mode == 'attention':
                with tf.variable_scope('attention'):
                    session_embed_mask = tf.sequence_mask(session_len, maxlen=self.group_max, dtype=tf.float32)
                    encoder_out, session_h = self.session_embed(
                        re_embedding, session_alias, session_mask, session_len, user, train=train)
                    decoder_input = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], 0)
                    dec_mask = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    decoder_out = trans_attention(tf.reshape(encoder_out, [self.batch_size, -1, self.hidden_size]),
                                                  tf.reshape(decoder_input, [self.batch_size, -1, self.hidden_size]),
                                                  session_embed_mask, dec_mask, self.hidden_size)
                    # stamp
                    decoder_last = tf.gather_nd(decoder_out,
                                                tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask - 1],
                                                         axis=1))
                    ma = stamp_attention(decoder_out, decoder_last, dec_mask, self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        ma = tf.concat([ma, session_h], -1)

            elif self.mode == 'usual_attention':
                with tf.variable_scope('usual_attention'):
                    seq_mask_ = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    seq_h = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], axis=0)
                    #加入self attention
                    if self.decoder_attention:
                        seq_h = multihead_attention(tf.reshape(seq_h, [self.batch_size, -1, self.hidden_size]), seq_mask_,
                                                    tf.reshape(seq_h, [self.batch_size, -1, self.hidden_size]), seq_mask_,
                                                    self.hidden_size, causality=False, scope='self_attention')
                    last_h = tf.gather_nd(seq_h, tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask-1], axis=1))
                    ma = stamp_attention(seq_h, last_h, seq_mask_,self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        session_re = self.session_embed(
                            re_embedding, session_alias, session_mask, session_len, user, train=train)
                        ma = tf.concat([ma, session_re], -1)
            if self.user_:
                user_embed = tf.nn.embedding_lookup(self.user_embedding, user)
                ma = tf.concat([ma, user_embed], -1)
                self.B = tf.get_variable('B', [self.control_dim * self.hidden_size + self.user_size, self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            else:
                self.B = tf.get_variable('B', [self.control_dim * self.hidden_size, self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tar-1, logits=logits))
            if train:
                self.vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name
                                   not in ['b_in', 'b_out', 'u_b', 'h_b']]) * self.L2
                train_loss = loss + lossL2
                self.opt = tf.train.AdamOptimizer(self.lr).minimize(train_loss)
                return train_loss, self.opt

            else:
                index = tf.nn.top_k(logits, 20)
                return loss, index

    def forward_test(self, adj_in, adj_out, items, seq_alias, seq_mask,
                     session_alias, session_len, session_mask, tar, user, num_item):
        train = False

        if self.graph == 'ggnn':
            re_embedding = self.ggnn(items, user, adj_in, adj_out, is_training=train)
        elif self.graph == 'no_graph':
            re_embedding = self.no_graph(items)
        b = self.embedding[1:]
        with tf.variable_scope('forward'):
            if self.mode == 'transformer':
                with tf.variable_scope('transformer'):
                    session_embed_mask = tf.sequence_mask(session_len, maxlen=self.group_max, dtype=tf.float32)
                    encoder_out, session_h = self.session_embed(
                        re_embedding, session_alias, session_mask, session_len, user, train=train)
                    decoder_input = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], 0)
                    dec_mask = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    decoder_out = mul_attention(
                        tf.reshape(decoder_input, [self.batch_size, -1, self.hidden_size]),
                        dec_mask,
                        tf.reshape(encoder_out, [self.batch_size, -1, self.hidden_size]),
                        session_embed_mask,
                        self.hidden_size, data=self.data)
                    #stamp
                    decoder_last = tf.gather_nd(decoder_out,
                                                tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask-1], axis=1))
                    ma = stamp_attention(decoder_out, decoder_last,  dec_mask, self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        ma = tf.concat([ma, session_h], -1)
            elif self.mode == 'attention':
                with tf.variable_scope('attention'):
                    session_embed_mask = tf.sequence_mask(session_len, maxlen=self.group_max, dtype=tf.float32)
                    encoder_out, session_h = self.session_embed(
                        re_embedding, session_alias, session_mask, session_len, user, train=train)
                    decoder_input = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], 0)
                    dec_mask = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    decoder_out = trans_attention(tf.reshape(encoder_out, [self.batch_size, -1, self.hidden_size]),
                                                  tf.reshape(decoder_input, [self.batch_size, -1, self.hidden_size]),
                                                  session_embed_mask, dec_mask, self.hidden_size)
                    # stamp
                    decoder_last = tf.gather_nd(decoder_out,
                                                tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask - 1],
                                                         axis=1))
                    ma = stamp_attention(decoder_out, decoder_last, dec_mask, self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        ma = tf.concat([ma, session_h], -1)

            elif self.mode == 'usual_attention':
                with tf.variable_scope('usual_attention'):
                    seq_mask_ = tf.sequence_mask(seq_mask, maxlen=seq_alias.shape[-1].value, dtype=tf.float32)
                    seq_h = tf.stack(
                        [tf.nn.embedding_lookup(re_embedding[i], seq_alias[i]) for i in range(self.batch_size)], axis=0)
                    #加入self attention
                    if self.decoder_attention:
                        seq_h = multihead_attention(tf.reshape(seq_h, [self.batch_size, -1, self.hidden_size]), seq_mask_,
                                                    tf.reshape(seq_h, [self.batch_size, -1, self.hidden_size]), seq_mask_,
                                                    self.hidden_size, causality=False, scope='self_attention')
                    last_h = tf.gather_nd(seq_h, tf.stack([tf.range(self.batch_size, dtype=tf.int64), seq_mask-1], axis=1))
                    ma = stamp_attention(seq_h, last_h, seq_mask_,self.u_w1, self.u_w2, self.u_b, self.u_v,
                                         self.hidden_size, self.batch_size)
                    if self.history_:
                        session_re = self.session_embed(
                            re_embedding, session_alias, session_mask, session_len, user, train=train)
                        ma = tf.concat([ma, session_re], -1)
            if self.user_:
                user_embed = tf.nn.embedding_lookup(self.user_embedding, user)
                ma = tf.concat([ma, user_embed], -1)
                self.B = tf.get_variable('B', [self.control_dim * self.hidden_size + self.user_size, self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            else:
                self.B = tf.get_variable('B', [self.control_dim * self.hidden_size, self.hidden_size],
                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tar-1, logits=logits))
            #print("num_item:" + str(num_item))
            index = tf.nn.top_k(logits, k=num_item-1)
            return loss, index, items

class Graph(Model):
    def __init__(self, hidden_size, user_size=10, batch_size=100, seq_max=20, group_max=50, n_item=None, n_user=None,
                 n_behaviour=None, lr=None, l2=None, step=1, decay=None, spare=True, ggnn_drop=0,
                 graph='ggnn', mode='usual_attention', data=None,
                 decoder_attention=True, encoder_attention=True, user_=True, behaviour_=False, history_=True, pool='max'):
        super(Graph, self).__init__(hidden_size, user_size, batch_size, seq_max, group_max, mode, data,
                                    decoder_attention, encoder_attention, user_,
                                    behaviour_, history_, spare)
        self.item_embedding = tf.get_variable(shape=[n_item-1, hidden_size], name='embedding', dtype=tf.float32,
                                              initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.embedding = tf.concat([tf.constant([-1e10], shape=[1, self.hidden_size]), self.item_embedding], 0)
        if self.user_:
            self.user_embedding = tf.get_variable(shape=[n_user, user_size], name='user_embedding', dtype=tf.float32,
                                                 initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        if self.behaviour_:
            self.b_embedding = tf.get_variable(shape=[n_behaviour, hidden_size], name='behaviour_embedding',
                                               dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.n_item = n_item
        self.n_user = n_user
        self.n_behaviour = n_behaviour
        self.graph = graph
        self.L2 = l2
        self.lr = lr
        self.ggnn_drop = ggnn_drop
        self.step = step
        self.decay = decay
        self.pool = pool
        if self.graph == 'ggnn' or 'gcn':
            self.W_in = tf.get_variable('W_in', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.b_in = tf.get_variable('b_in', [self.hidden_size], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            self.W_out = tf.get_variable('W_out', [self.hidden_size, self.hidden_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

            self.b_out = tf.get_variable('b_out', [self.hidden_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            if self.user_:
                self.u_in = tf.get_variable('u_in', shape=[user_size, self.hidden_size], dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
                self.u_out = tf.get_variable('u_out', [user_size, self.hidden_size], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

    def ggnn(self, items, user, adj_in, adj_out, is_training=True):
        fin_state = tf.nn.embedding_lookup(self.embedding, items)
        u_state = tf.nn.embedding_lookup(self.user_embedding, user)
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        with tf.variable_scope('GGNN'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [-1, self.hidden_size])
                if self.user_:
                    fin_state_in = tf.reshape(tf.matmul(fin_state, self.W_in)+self.b_in,
                                              [self.batch_size, -1, self.hidden_size]) + \
                                   tf.expand_dims(tf.matmul(u_state, self.u_in), 1)
                    fin_state_out = tf.reshape(tf.matmul(fin_state, self.W_out) + self.b_out,
                                              [self.batch_size, -1, self.hidden_size]) + \
                                    tf.expand_dims(tf.matmul(u_state, self.u_out), 1)
                else:
                    fin_state_in = tf.reshape(tf.matmul(fin_state, self.W_in) + self.b_in,
                                              [self.batch_size, -1, self.hidden_size])
                    fin_state_out = tf.reshape(tf.matmul(fin_state, self.W_out) + self.b_out,
                                               [self.batch_size, -1, self.hidden_size])

                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)

                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.hidden_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.hidden_size]))
                #fin_state = tf.layers.dropout(fin_state, rate= self.ggnn_drop, training=tf.convert_to_tensor(is_training))
        return tf.reshape(fin_state, [self.batch_size, -1, self.hidden_size])

    def no_graph(self, items):
        fin_state = tf.nn.embedding_lookup(self.embedding, items)
        return fin_state


def variable_summaries(var, name):
    tf.summary.histogram(name,var)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/'+name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar('stddev/'+name, stddev)


def stamp_attention(seq, last, seq_mask, w1, w2, b, v, dim, batchsize):
    """
    :param seq: N*T*D
    :param last:  N*D
    :param seq_mask: N*T
    :return: 2*D
    """
    last = tf.matmul(last, w1)
    seq_h = tf.matmul(tf.reshape(seq, [-1, dim]), w2)
    m = tf.nn.sigmoid(tf.expand_dims(last, 1)+tf.reshape(seq_h, [batchsize, -1, dim])+ b)
    coef = tf.matmul(tf.reshape(m, [-1, dim]), v, transpose_b=True)*tf.reshape(seq_mask, [-1,1])
    ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [batchsize, -1, 1])*seq, 1), tf.reshape(last, [-1, dim])], -1)
    return ma


def user_attention(seq, user, seq_mask, w, b, dim, batchsize):
    """
    :param seq:   N*T*D
    :param user:  N*D
    :param seq_mask: N*T
    :param w: D*D
    :param b: D
    :param dim: D
    :param batchsize: N
    :return: N*D
    """
    seq_ = tf.reshape(tf.matmul(tf.reshape(seq,[-1,dim]), w), [batchsize,-1,dim])
    coef=tf.squeeze(tf.matmul(seq_, tf.expand_dims(user,2)))*seq_mask
    padding = tf.ones_like(seq_mask)*(-2**32+1)
    coef = tf.where(tf.equal(coef, 0), padding, coef)
    coef = tf.nn.softmax(coef)
    out = tf.squeeze(tf.reduce_sum(seq*tf.expand_dims(coef,2),1))
    return out


# reference paper------neural machine translation by jointly learning to align and translate-----------
def trans_attention(sess, seq, session_mask, seq_mask, dim):
    sess_ = tf.layers.dense(sess, dim, activation=None, use_bias=False, name='sess_')
    seq_ = tf.layers.dense(seq, dim, activation=None, use_bias=False, name='seq_')
    #coef = tf.squeeze(tf.layers.dense((tf.expand_dims(seq_, 2) + tf.expand_dims(sess_, 1)), 1, activation=tf.nn.tanh, use_bias=False, name='coef'))
    coef = tf.squeeze(
        tf.layers.dense(tf.nn.tanh(tf.expand_dims(seq_, 2) + tf.expand_dims(sess_, 1)), 1, activation=None, use_bias=False,
                        name='coef'))
    sess_masks = tf.tile(tf.expand_dims(session_mask, 1), [1, tf.shape(seq)[1], 1])
    paddings = tf.ones_like(coef)*(-2**32+1)
    outputs = tf.where(tf.equal(sess_masks, 0), paddings, coef)
    outputs = tf.nn.softmax(outputs)
    seq_masks = tf.tile(tf.expand_dims(seq_mask, -1), [1, 1, tf.shape(sess)[1]])
    outputs = tf.layers.dense(tf.concat([tf.matmul(outputs*seq_masks, sess), seq], axis=-1), dim, activation=None, use_bias=False,name='concat')
    outputs = normalize(outputs)
    return outputs


def mul_attention(queries, query_masks, keys, key_masks, dim, data='xing'):
    with tf.variable_scope('multihead_attention'):
        Q = tf.layers.dense(queries, dim, activation=tf.nn.relu, use_bias=False, name='q')  # (N, T_q, C)
        K = tf.layers.dense(keys, dim, activation=tf.nn.relu, use_bias=False, name='k')  # (N, T_k, C)
        V = tf.layers.dense(keys, dim, activation=tf.nn.relu, use_bias=False, name='v')  # (N, T_k, C)

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks
        #outputs = tf.layers.dense(tf.concat([tf.matmul(outputs, V), queries],axis=-1), dim, activation=None, use_bias=False, name='concat')
        outputs = tf.matmul(outputs, V)+queries
        if data == 'xing':
            outputs = normalize(outputs)
            return outputs
        else:
            return outputs


def parse_function_(max_session):
    def parse_function(example_proto):
        dics = {'tar': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                # when parse the example, shape below can be used as reshape, for example reshape (3,) to (1,3)
                'user': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'session_alias': tf.VarLenFeature(dtype=tf.int64),
                'session_alias_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'session_mask': tf.FixedLenFeature(shape=(max_session,), dtype=tf.int64),
                'session_len': tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'seq_alias': tf.VarLenFeature(dtype=tf.int64),
                'seq_mask':tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'num_node':tf.FixedLenFeature(shape=(), dtype=tf.int64),
                'all_node':tf.VarLenFeature(dtype=tf.int64),
                # we can use VarLenFeature, but it returns SparseTensor
                # 'A_in': tf.VarLenFeature(dtype=tf.float32),
                # 'A_in_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                # 'A_out': tf.VarLenFeature(dtype=tf.float32),
                # 'A_out_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                #------------稀疏-------------------
                'A_in_row': tf.VarLenFeature(dtype=tf.int64),
                'A_in_col': tf.VarLenFeature(dtype=tf.int64),
                'A_in': tf.VarLenFeature(dtype=tf.float32),
                'A_out_row': tf.VarLenFeature(dtype=tf.int64),
                'A_out_col': tf.VarLenFeature(dtype=tf.int64),
                'A_out': tf.VarLenFeature(dtype=tf.float32),
                'A_in_shape':tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'A_out_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
                #-----------二进制保存-------------------
                # 'A_in': tf.FixedLenFeature([], tf.string),
                # 'A_out': tf.FixedLenFeature([], tf.string)
                #-----------------------------
                }
        parsed_example = tf.parse_single_example(example_proto, dics)
        parsed_example['session_alias'] = tf.sparse_tensor_to_dense(parsed_example['session_alias'])
        parsed_example['session_alias'] = tf.reshape(parsed_example['session_alias'], parsed_example['session_alias_shape'])
        parsed_example['all_node'] = tf.sparse_tensor_to_dense(parsed_example['all_node'])
        parsed_example['seq_alias'] = tf.sparse_tensor_to_dense(parsed_example['seq_alias'])
        #-------------正常方式----------------------
        # parsed_example['A_in'] = tf.sparse_tensor_to_dense(parsed_example['A_in'])
        # parsed_example['A_in'] = tf.reshape(parsed_example['A_in'], parsed_example['A_in_shape'])
        # parsed_example['A_out'] = tf.sparse_tensor_to_dense(parsed_example['A_out'])
        # parsed_example['A_out'] = tf.reshape(parsed_example['A_out'], parsed_example['A_out_shape'])
        #--------------稀疏方式----------
        parsed_example['A_in_row'] = tf.sparse_tensor_to_dense(parsed_example['A_in_row'])
        parsed_example['A_in_col'] = tf.sparse_tensor_to_dense(parsed_example['A_in_col'])
        parsed_example['A_out_row'] = tf.sparse_tensor_to_dense(parsed_example['A_out_row'])
        parsed_example['A_out_col'] = tf.sparse_tensor_to_dense(parsed_example['A_out_col'])
        parsed_example['A_in'] = tf.sparse_tensor_to_dense(parsed_example['A_in'])
        parsed_example['A_out'] = tf.sparse_tensor_to_dense(parsed_example['A_out'])
        parsed_example['A_in'] =\
            tf.SparseTensor(indices=tf.transpose(tf.stack([parsed_example['A_in_row'], parsed_example['A_in_col']])),
                            values=parsed_example['A_in'], dense_shape=parsed_example['A_in_shape'])
        parsed_example['A_out'] = \
            tf.SparseTensor(indices=tf.transpose(tf.stack([parsed_example['A_out_row'], parsed_example['A_out_col']])),
                            values=parsed_example['A_out'], dense_shape=parsed_example['A_out_shape'])
        parsed_example['A_in'] = tf.sparse_tensor_to_dense(parsed_example['A_in'])
        parsed_example['A_out'] = tf.sparse_tensor_to_dense(parsed_example['A_out'])
        #-------------二进制读取----------------
        # parsed_example['A_in'] = tf.decode_raw(parsed_example['A_in'], tf.float32)
        # parsed_example['A_out'] = tf.decode_raw(parsed_example['A_out'], tf.float32)
        #-----------------------------------------
        return parsed_example
    return parse_function


def run_epoch(session, train_loss, train_opt, step, max_length, max_session):
    loss = []
    while True:
        try:
            loss_, _ = session.run([train_loss, train_opt])
            loss.append(loss_)
            step += 1
            #if step%5000 == 0:
                #session.run(valid_iterator.initializer)
                #val_loss, hit5, hit10, hit20, mrr5, mrr10, mrr20, _,_ = eval_epoch(session, valid_index, valid_loss, valid_data, max_length=max_length, max_session=max_session)
                #print('---After %d steps' % (step),
                #      'train_loss:%.4f\tvalid_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tMMR@5:%.4f'
                #      '\tMrr@10:%.4f\tMMR@20:%.4f' %(loss_, val_loss, hit5, hit10, hit20, mrr5, mrr10, mrr20))
        except tf.errors.OutOfRangeError:
            break
    return step, np.mean(loss)


def eval_epoch(session, test_index, test_loss, test_data, max_length=20, max_session=150):
    all_loss, hit5, mrr5, hit10, mrr10, hit20, mrr20 = [], [], [], [], [], [], []
    length_index = np.zeros((max_length-1, 8))
    history_index = np.zeros((max_session, 8))
    length_index[:,6] = length_index[:,6] + 1
    history_index[:,6] = history_index[:,6] + 1
    while True:
        try:
            index, test_loss_, tar, seq_length, sess_length = session.run([test_index, test_loss, test_data['tar'], test_data['seq_mask'], test_data['session_len']])
            all_loss.append(test_loss_)
            for score, target, length, length_ in zip(index[1], tar, seq_length, sess_length):
                hit20.append(np.isin(target - 1, score))
                length_index[length-1, 2] += np.isin(target - 1, score)
                history_index[length_-1, 2] += np.isin(target - 1, score)
                hit10.append(np.isin(target-1,  score[0:10]))
                length_index[length - 1, 1] += np.isin(target - 1, score[0:10])
                history_index[length_-1, 1] += np.isin(target - 1, score[0:10])
                hit5.append(np.isin(target - 1, score[0:5]))
                length_index[length - 1, 0] += np.isin(target - 1, score[0:5])
                history_index[length_ - 1, 0] += np.isin(target - 1, score[0:5])
                length_index[length-1, 6] += 1
                history_index[length_- 1, 6] += 1
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
                    length_index[length - 1, 5] += 1 / (np.where(score == target - 1)[0][0] + 1)
                    history_index[length_ - 1, 5] += 1 / (np.where(score == target - 1)[0][0] + 1)
                if len(np.where(score[0:10] == target - 1)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score[0:10] == target - 1)[0][0] + 1))
                    length_index[length - 1, 4] += 1 / (np.where(score == target - 1)[0][0] + 1)
                    history_index[length_ - 1, 4] += 1 / (np.where(score == target - 1)[0][0] + 1)
                if len(np.where(score[0:5] == target - 1)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score[0:5] == target - 1)[0][0] + 1))
                    length_index[length - 1, 3] += 1 / (np.where(score == target - 1)[0][0] + 1)
                    history_index[length_ - 1, 3] += 1 / (np.where(score == target - 1)[0][0] + 1)
        except tf.errors.OutOfRangeError:
            break
    #length_index = length_index.cumsum(0)
    #history_index = history_index.cumsum(0)
    for i in range(6):
        length_index[:, i] = length_index[:, i] / length_index[:, 6]
        history_index[:, i] = history_index[:, i] / history_index[:, 6]
    length_index[:, -1] = np.arange(1, max_length)
    history_index[:, -1] = np.arange(1, max_session+1)
    # len_index = pd.DataFrame(length_index,
    #                          columns = ['RecaLL5', 'RecaLL10', 'RecaLL20', 'Mrr5', 'Mrr10', 'Mrr20', 'number'],
    #                          index=range(1, max_length))
    return np.mean(all_loss), np.mean(hit5)*100, np.mean(hit10)*100, np.mean(hit20)*100, \
           np.mean(mrr5)*100, np.mean(mrr10)*100, np.mean(mrr20)*100, length_index, history_index

#生成训练集
def random_name(path):
    train_filenames = tf.train.match_filenames_once(path)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        filename = sess.run(train_filenames)
        for i in range(5):
            random.shuffle(filename)
    return filename

#从测试集中选出部分作为验证集
def random_validation(test_path):
    train_filenames = tf.train.match_filenames_once(test_path)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        filename = sess.run(train_filenames)
        length = len(filename)
        vali_file = np.random.choice(filename, int(0.1*length))
    return vali_file


def shulle_train(path, max_session, buffer_size, padded_shape, batchSize):
    train_filenames = tf.train.match_filenames_once(path)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        filename = sess.run(train_filenames)
        for i in range(3):
            random.shuffle(filename)
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(parse_function_(max_session)).shuffle(buffer_size=buffer_size)
    train_batch_padding_dataset = train_dataset.padded_batch(batchSize, padded_shapes=padded_shape,
                                                             drop_remainder=True)
    train_iterator = train_batch_padding_dataset.make_initializable_iterator()
    train_data = train_iterator.get_next()

    return train_iterator, train_data


