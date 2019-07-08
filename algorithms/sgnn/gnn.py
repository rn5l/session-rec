#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : gnn.py
# @Software: PyCharm
import tensorflow as tf
import math
from algorithms.sgnn.utils import Data, prepare_data
import numpy as np
import datetime
import pandas as pd


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

    def forward(self, re_embedding, batch_size, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(tf.reshape(coef, [batch_size, -1, 1]) * seq_h, 1)
            logits = tf.matmul(ma, b, transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, feed_dic):
        return self.sess.run(fetches, feed_dic)


class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=100,
                 lr=0.001, l2=0.00001, step=1, lr_dc=0.1, lr_dc_step=3, nonhybrid=True, epoch_n=30, batch_predict=False):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)

        self.L2 = l2
        self.step = step
        self.lr_dc_step = lr_dc_step
        self.nonhybrid = nonhybrid
        self.lr_dc = lr_dc        
        self.lr = lr
        self.epoch_n = epoch_n

        # updated while recommending
        self.session = -1
        self.session_items = []

        # for case batch_predict=True
        self.batch_predict = batch_predict
        self.test_idx = 0
    
    def init_model(self):
        
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.tar = tf.placeholder(dtype=tf.int32)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        
        self.embedding = tf.get_variable(shape=[self.n_nodes, self.hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in_tr = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out_tr = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        if self.batch_predict:
            self.adj_in_ts = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
            self.adj_out_ts = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        else:
            self.adj_in_ts = tf.placeholder(dtype=tf.float32, shape=[1, None, None])
            self.adj_out_ts = tf.placeholder(dtype=tf.float32, shape=[1, None, None])
        
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        
        with tf.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn(self.batch_size, self.adj_in_tr, self.adj_out_tr),self.batch_size)
        with tf.variable_scope('ggnn_model', reuse=True):
            if self.batch_predict:
                self.loss_test, self.score_test = self.forward(self.ggnn(self.batch_size, self.adj_in_ts, self.adj_out_ts),self.batch_size, train=False)
            else:
                self.loss_test, self.score_test = self.forward(self.ggnn(1, self.adj_in_ts, self.adj_out_ts), 1, train=False)
            
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=self.decay,
                                                        decay_rate=self.lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    
    def ggnn(self, batch_size, adj_in, adj_out):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [batch_size, -1, self.out_size])

    def fit(self, train, test=None, sample_store=10000000):

        method = 'ggnn'
        
        self.n_nodes = len( train.ItemId.unique() ) + 1

        train_data, test_data, self.item_dict, self.reversed_item_dict = prepare_data(train, test)
        
        self.train_data = Data(train_data, sub_graph=True, method=method, shuffle=True)
        self.test_data = Data(test_data, sub_graph=True, method=method, shuffle=False)
        
        self.decay = self.lr_dc_step * len(self.train_data.inputs) / self.batch_size
        
        with tf.Graph().as_default():
        
            self.init_model()
            
            for epoch in range(self.epoch_n):
            # print('epoch: ', epoch, '===========================================')
                slices = self.train_data.generate_batch(self.batch_size)
                fetches = [self.opt, self.loss_train, self.global_step]
                print('start training: ', datetime.datetime.now())
                for i, j in zip(slices, np.arange(len(slices))):
                    adj_in, adj_out, alias, item, mask, targets = self.train_data.get_slice(i)
                    feed_dict = {self.tar: targets, self.item: item, self.adj_in_tr: adj_in,
                                 self.adj_out_tr: adj_out, self.alias: alias, self.mask: mask}
                    _, loss, _ = self.run(fetches, feed_dict)
                
                
                if self.batch_predict:
                    slices = self.test_data.generate_batch(self.batch_size)
                else:
                    slices = self.test_data.generate_batch(1)
                print('start predicting: ', datetime.datetime.now())
                hit, mrr, test_loss_ = [], [],[]


                all_scores = None

                for i, j in zip(slices, np.arange(len(slices))):

                    adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice(i)
                    feed_dict = {self.tar: targets, self.item: item, self.adj_in_ts: adj_in,
                                 self.adj_out_ts: adj_out, self.alias: alias, self.mask: mask}
                    scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
                    test_loss_.append(test_loss)

                    if self.batch_predict:
                        if all_scores is None:
                            all_scores = scores
                        else:
                            all_scores = np.concatenate( [all_scores, scores] )
                        self.all_scores = all_scores

                    index = np.argsort(scores, 1)[:, -20:]
                    for score, target in zip(index, targets):
                        hit.append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
                hit = np.mean(hit) * 100
                mrr = np.mean(mrr) * 100
                test_loss = np.mean(test_loss_)
                print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d' %
                      (loss, test_loss, hit, mrr, epoch))

            if self.batch_predict:
                self.predicted_item_ids = []
                for idx in range(len(self.all_scores[0])):
                    self.predicted_item_ids.append(int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)


    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()

        # convert original item_id according to the item_dic
        item_id_dic = self.item_dict[input_item_id]
        self.session_items.append(item_id_dic)

        if self.batch_predict:
            scores = self.all_scores[self.test_idx]
            self.test_idx += 1

        else:
            adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice_by_session_items(self.session_items, self.test_data.len_max)
            feed_dict = {self.tar: targets, self.item: item, self.adj_in_ts: adj_in,
                         self.adj_out_ts: adj_out, self.alias: alias, self.mask: mask}
            scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
            self.predicted_item_ids = []
            scores = scores[0]
            for idx in range(len(scores)):
                self.predicted_item_ids.append(int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)


        series = pd.Series(data=scores, index=self.predicted_item_ids)
        return series

    def clear(self):
        self.sess.close()
        pass