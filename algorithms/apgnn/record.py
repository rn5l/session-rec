#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/12/3 3:37
# @Author : {ZM7}
# @File : save_tfrecord.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import pickle
import argparse
import os
from scipy import sparse as sp
import datetime
import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
from .model_last import parse_function_

lastfm_path = './datasets/3_user_sessions.pickle'
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='xing', help='data name: xing/last')
parser.add_argument('--dataset', default='sample', help='dataset name: all_data/sample')
parser.add_argument('--user', type=int, default=50, help='the number of need user')
parser.add_argument('--graph', default='ggnn')
parser.add_argument('--adj', default='adj', help='adj_all')
parser.add_argument('--max_session', type=int, default=100)
parser.add_argument('--max_length', type=int, default=20)
parser.add_argument('--last', action='store_true', help='user_embedding')
parser.add_argument('--bpr', action='store_true', help='cross entropy/bpr')

opt = parser.parse_args()

if opt.last:
    train_path = './datasets/' + opt.data + '/'+ opt.graph + '/' + 'tfrecord_' + str(
        opt.max_session) + '_' + str(opt.max_length) + '_' + opt.adj + '_last' + '/' + opt.dataset
    test_path = './datasets/' + opt.data + '/' + opt.graph + '/' + 'tfrecord_' + str(
        opt.max_session) + '_' + str(opt.max_length) + '_' +opt.adj + '_last' + '/' + opt.dataset
else:
    train_path = './datasets/' + opt.data + '/'+ opt.graph+'/'+'tfrecord_'+str(opt.max_session)\
                 + '_' + str(opt.max_length) +'_'+opt.adj+'/'+opt.dataset
    test_path = './datasets/' + opt.data + '/'+ opt.graph+'/'+'tfrecord_'+str(opt.max_session)\
                + '_' + str(opt.max_length) +'_'+opt.adj+'/'+opt.dataset


#定义整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#定义浮点列表型的属性
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=value))
#定义生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def mkdir(path):
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return
    else:
        return


def apply_parallel(df_grouped, func):
    Parallel(n_jobs=8)(delayed(func)(group)for name, group in df_grouped)


def generate_tfrecord(all_data, train_path, graph='ggnn', max_session=50, max_length=20,
                test_size=0.2, adj='adj', last=True, user_number=None):
    if user_number:
        user_sample = np.random.choice(all_data['user'].unique(), user_number)
        all_data = all_data[all_data['user'].isin(user_sample)]

    def select_data(train=True):
        def select(data):
            orgin_path = train_path+'/user_'+str(np.unique(data['user'])[0])+'/'
            mkdir(orgin_path)
            all_sess = data['session_id'].unique()
            if last:
                split_point = len(all_sess)-1
            else:
                split_point = int(len(all_sess)*(1-test_size))
            user_id = data['user'].unique()[0]
            count = 1
            if len(all_sess) == 1:
                return None
            if train:
                orgin_path = orgin_path+'train_'
                start = 1
                end = split_point
            else:
                # writer = tf.python_io.TFRecordWriter(
                #     test_path + '/' + 'test_user_' + str(np.unique(data['user_id'])[0]) + '.tfrecord')
                orgin_path = orgin_path + 'test_'
                start = split_point
                end = len(all_sess)
            writer = tf.python_io.TFRecordWriter(orgin_path + str(count)+'.tfrecord')
            for i in range(start, end):
                #生成session和seq
                if i < max_session + 1:
                    all_seq = data[data['session_id'] == all_sess[i]]['item'].values.tolist()
                    sub_sess = [data[data['session_id'] == sess]['item'].values.tolist() for sess in all_sess[0:i]]
                else:
                    all_seq = data[data['session_id'] == all_sess[i]]['item'].values.tolist()
                    sub_sess = [data[data['session_id'] == sess]['item'].values.tolist() for sess in all_sess[i-max_session:i]]
                sub_node = np.hstack(sub_sess)
                for j in range(len(all_seq) - 1):
                    features = {}
                    sub_seq = all_seq[0:j + 1]
                    features['tar'] = _int64_feature([all_seq[j + 1]])
                    features['user'] = _int64_feature([user_id])
                    node = np.unique(np.hstack([sub_node, sub_seq, [0]]))
                    #生成每个session的别名和mask值，并且padding
                    sub_sess_pad = [sess + [0]*(max_length-len(sess)) for sess in sub_sess]+[[0]*max_length]*(max_session-len(sub_sess))
                    sub_sess_alias = np.array([[np.where(node==s)[0][0] for s in sess_pad] for sess_pad in sub_sess_pad])
                    features['session_alias'] = _int64_feature(sub_sess_alias.reshape(-1))
                    features['session_alias_shape'] = _int64_feature(sub_sess_alias.shape)
                    #session mask值
                    features['session_mask'] =_int64_feature([len(sess) for sess in sub_sess]+[1]*(max_session-len(sub_sess)))
                    #session_len每个session序列中session的数量
                    features['session_len'] = _int64_feature([len(sub_sess)])
                    #生成seq别名和mask值并且padding
                    sub_seq_pad = sub_seq#+[0]*(max_length-len(sub_seq))
                    sub_seq_alias = [np.where(node == s)[0][0] for s in sub_seq_pad]
                    features['seq_alias'] = _int64_feature(sub_seq_alias)
                    #seq_pad.append(sub_seq_pad)
                    #seq mask值
                    features['seq_mask'] = _int64_feature([len(sub_seq)])
                    #节点数量
                    features['num_node'] = _int64_feature([len(node)])
                    features['all_node'] = _int64_feature(node)
                    if graph == 'ggnn':
                        u_A = np.zeros((len(node), len(node)))
                    elif graph == 'gcn':
                        u_A = np.eye(len(node))
                    for u_input in sub_sess:
                        for k in np.arange(len(u_input)-1):
                            u = np.where(node == u_input[k])[0][0]
                            v = np.where(node == u_input[k + 1])[0][0]
                            if adj == 'adj_all':
                                u_A[u][v] += 1
                            else:
                                u_A[u][v] = 1
                    for l in np.arange(len(sub_seq)-1):
                        u = np.where(node == sub_seq[l])[0][0]
                        v = np.where(node == sub_seq[l + 1])[0][0]
                        if adj == 'adj_all':
                            u_A[u][v] += 1
                        else:
                            u_A[u][v] = 1
                    u_sum_in = np.sum(u_A, 0)
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)
                    u_sum_out = np.sum(u_A, 1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)
                    #------------稀疏方式------------
                    u_A_in = sp.coo_matrix(u_A_in)
                    u_A_out = sp.coo_matrix(u_A_out)
                    features['A_in_row'] = _int64_feature(u_A_in.row)
                    features['A_in_col'] = _int64_feature(u_A_in.col)
                    features['A_in'] = _float_feature(u_A_in.data)
                    features['A_out_row'] = _int64_feature(u_A_out.row)
                    features['A_out_col'] = _int64_feature(u_A_out.col)
                    features['A_out'] = _float_feature(u_A_out.data)
                    features['A_in_shape'] = _int64_feature(u_A_in.shape)
                    features['A_out_shape'] = _int64_feature(u_A_out.shape)
                    #--------------------------------------
                    tf_features = tf.train.Features(feature=features)
                    tf_example = tf.train.Example(features=tf_features)
                    tf_serialized = tf_example.SerializeToString()
                    writer.write(tf_serialized)
                    count += 1
                    if count%200 == 0 and i != end-1:
                        writer.close()
                        writer = tf.python_io.TFRecordWriter(orgin_path + str(count) + '.tfrecord')
            writer.close()
        return select

    apply_parallel(all_data.groupby('user'), select_data())
    apply_parallel(all_data.groupby('user'), select_data(False))


if __name__ == '__main__':
    if opt.data == 'last':
        all_data=pd.read_csv(r'./datasets/last.csv')
    elif opt.data == 'xing':
        all_data = pd.read_csv(r'./datasets/xing.csv')
    elif opt.data == 'reddit':
        all_data = pd.read_csv(r'./datasets/reddit.csv')
    print('start:',datetime.datetime.now())
    if opt.dataset == 'sample':
        generate_tfrecord(all_data, train_path,  graph=opt.graph, max_session=opt.max_session, max_length=opt.max_length, adj=opt.adj,
                          last=opt.last, user_number=opt.user)
    else:
        generate_tfrecord(all_data, train_path, graph=opt.graph, max_session=opt.max_session, max_length=opt.max_length, adj=opt.adj, last=opt.last)
    print('end:', datetime.datetime.now())
