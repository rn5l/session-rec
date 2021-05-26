#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np
import operator
from _collections import OrderedDict


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
            missing = self.length % batch_size
            fill = batch_size - missing
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        
        if self.length % batch_size != 0:
            slices[-1] = np.concatenate( [ np.arange(self.length-missing, self.length), np.arange(self.length-fill, self.length) ])
        return slices

    def get_slice(self, index):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            if self.method == 'ggnn':
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.zeros((max_n_node, max_n_node))
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    u_sum_in = np.sum(u_A, 0)
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)
                    u_sum_out = np.sum(u_A, 1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)

                    A_in.append(u_A_in)
                    A_out.append(u_A_out)
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]

    def get_slice_by_session_items(self, session, max_len):
        items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []

        # predefined mask size:
        MASK_SIZE = max_len   #8

        # create an empty mask of pre-defined size and fill it
        innermask = np.zeros(MASK_SIZE, dtype = int)
        innermask[:len(session)] = 1
        mask = np.asarray([innermask])

        # transform the session into an ndarray
        session_array = np.zeros(MASK_SIZE, dtype = int)
        for i in range(len(session)):
            session_array[i] = session[i]

        # print('input: ' + str(session_array))
        # do original code, no for loop, needed.
        n_node.append(len(np.unique(session_array)))

        # could also remove unneded statements
        max_n_node = np.max(n_node)
        node = np.unique(session_array)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(session_array) - 1):
            if session_array[i + 1] == 0:
                break
            u = np.where(node == session_array[i])[0][0]
            v = np.where(node == session_array[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        A_in.append(u_A_in)
        A_out.append(u_A_out)
        alias_inputs.append([np.where(node == i)[0][0] for i in session_array])

        # create empty target array
        target = [1]
        target = np.asarray(target, dtype = int)
        return A_in, A_out, alias_inputs, items, mask, target

# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra(tra_sess, sess_clicks):
    item_dict = {}
    reversed_item_dict = {}
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                reversed_item_dict[item_ctr] = i
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)  # 43098, 37484
    return train_ids, train_dates, train_seqs, item_dict, reversed_item_dict

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes(tes_sess, sess_clicks, item_dict):
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


def process_seqs(iseqs, idates, max_len=None):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in reversed( range(1, len(seq)) ):
            tar = seq[-i]
            labs += [tar]
            if max_len:
                out_seqs += [seq[max(0, len(seq) - i - max_len):-i]]
            else:
                out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


def prepare_data(train, test, max_len=None):

    sess_clicks = OrderedDict()

    sess_date_tr = OrderedDict()
    ctr = 0
    curid = -1
    curdate = None
    for tr in train.itertuples():
        # for data in reader:
        sessid = tr.SessionId
        if curdate and not curid == sessid:
            # date = ''
            # if opt.dataset == 'yoochoose':
            #     date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            # else:
            #     date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            date = tr.Time  # SARA
            sess_date_tr[curid] = date
        curid = sessid
        # if opt.dataset == 'yoochoose':
        #     item = data['item_id']
        # else:
        #     item = data['item_id'], int(data['timeframe']) todo: timeframe ?!
        item = int(tr.ItemId)  # SARA
        # curdate = ''
        # if opt.dataset == 'yoochoose':
        #     curdate = data['timestamp']
        # else:
        #     curdate = data['eventdate']
        curdate = tr.Time  # SARA
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    # date = ''
    # if opt.dataset == 'yoochoose':
    #     date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    # else:
    #     date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    #     for i in list(sess_clicks):   #this use digenetica 'timeframe' to sort items of each session. sess_clicks will be list of item_id s which are sorted based on their timeframe
    #         sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
    #         sess_clicks[i] = [c[0] for c in sorted_clicks]
    # sess_date[curid] = date
    sess_date_tr[curid] = tr.Time

    sess_date_ts = OrderedDict()
    ctr = 0
    curid = -1
    curdate = None
    for index, ts in test.iterrows():
        sessid = ts.SessionId
        if curdate and not curid == sessid:
            date = ts.Time
            sess_date_ts[curid] = date
        curid = sessid
        item = ts.ItemId
        curdate = ts.Time
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    sess_date_ts[curid] = ts.Time

    tra_sess = list(sess_date_tr.items())
    tes_sess = list(sess_date_ts.items())

    # Sort sessions by date MALTE: not needed
#     tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
#     tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]


    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    tra_ids, tra_dates, tra_seqs, item_dict, reversed_item_dict = obtian_tra(tra_sess, sess_clicks)
    tes_ids, tes_dates, tes_seqs = obtian_tes(tes_sess, sess_clicks, item_dict)

    tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, max_len=max_len)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, max_len=max_len)

    tra = (tr_seqs, tr_labs)
    tes = (te_seqs, te_labs)

    return tra, tes, item_dict, reversed_item_dict
