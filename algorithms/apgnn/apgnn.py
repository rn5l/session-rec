import numpy as np
import pandas as pd
import os
from scipy import sparse as sp
import tensorflow as tf
from joblib import Parallel, delayed
from .train_last import Trainer
import multiprocessing
from .model_last import parse_function_
import shutil


class apgnn:
    def __init__(self,
                 graph = 'ggnn', adj='adj',
                 max_session=30,
                 buffer_size=10000,
                 ggnn_drop=0.0,
                 epoch=30,
                 batchSize=10,
                 hiddenSize=100,
                 userSize=50,
                 decay=None,
                 l2=0.0,
                 lr=0.001,
                 mode='transformer',
                 decoder_attention=False,
                 encoder_attention=False,
                 user_=False,
                 history_=False,
                 behaviour_=False,
                 pool='max',
                 step=1,
                 session_key='SessionId', item_key='ItemId', user_key ='UserId', time_key='Time'):
        self.opt = Opt(max_session = max_session, buffer_size = buffer_size, ggnn_drop = ggnn_drop , adj = adj , epoch = epoch, batchSize = batchSize, hiddenSize = hiddenSize, userSize = userSize, decay = decay, l2 = l2 , lr = lr , graph = graph , mode = mode , decoder_attention = decoder_attention, encoder_attention = encoder_attention, user_ = user_, history_ = history_, behaviour_ = behaviour_, pool = pool, step = step)

        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key
        # TODO: ask to Sara best path. Maybe algorithm/filemodel/temp? or apgnn/temp?
        self.train_path = './temp/tfrecord'
        rmdir(self.train_path)
        self.test_path = './temp/tfrecord/test/'
        print("init")

        self.padded_shape = {'A_in': [None, None],
                        'A_out': [None, None],
                        'session_alias_shape': [None],
                        'session_alias': [None, None],
                        'seq_mask': [],
                        'session_len': [],
                        'tar': [],
                        'user': [],
                        'session_mask': [None],
                        'seq_alias': [None],
                        'num_node': [],
                        'all_node': [None],
                        'A_in_shape': [None],
                        'A_out_shape': [None],
                        'A_in_row': [None],
                        'A_in_col': [None],
                        'A_out_row': [None],
                        'A_out_col': [None]}


    def get_tensorflow_session(self):
        sess = tf.Session()
        tf.local_variables_initializer().run(session=sess)
        sess.run(tf.compat.v1.global_variables_initializer())
        return sess

    def fit(self, train_data, test_data=None):

        config = {
            'session_key': 'session_id',
            'item_key': 'item',
            'time_key': 'ts',
            'user_key': 'user',
            'train_path': self.train_path
        }

        max_train_length = train_data.groupby([self.session_key])[self.item_key].count().max()
        max_test_length = test_data.groupby([self.session_key])[self.item_key].count().max()
        self.max_length = max(max_train_length, max_test_length)

        self.itemids = train_data[self.item_key].unique()
#        print("train data items: " + str(len(self.itemids)))

        self.userids = train_data[self.user_key].unique()

        self.num_items = len(self.itemids) + 1
        self.item2id = dict(zip(self.itemids, range(1, len(self.itemids) + 1)))
        self.user2id = dict(zip(self.userids, range(1, len(self.userids) + 1)))

        self.id2item = dict()
        for k in self.item2id.keys():
            self.id2item[self.item2id[k]] = k

        self.num_users = len(self.user2id)

        train_data_new = train_data.loc[:, ['index', 'Time', 'UserId', 'ItemId', 'SessionId']]
        train_data_new = train_data_new.rename(columns={'index':0, 'Time':'ts', 'UserId':'user', 'ItemId':'item', 'SessionId':'session_id'})

        train_data_new['item'] = train_data_new['item'].apply(lambda x: self.item2id[x])
        train_data_new['user'] = train_data_new['user'].apply(lambda x: self.user2id[x])


        self.opt.set_max_length(self.max_length)

        self.train_data = train_data_new
        generate_tfrecord(train_data_new, config, graph=self.opt.graph, max_session=self.opt.max_session, max_length=self.opt.max_length,
                          adj=self.opt.adj)


        self.trainer = Trainer(self.opt, self.num_items, self.num_users)
        self.sess = self.get_tensorflow_session()
        self.trainer.train(self.sess)


        test_data_new = test_data.loc[:, ['index', 'Time', 'UserId', 'ItemId', 'SessionId']]
        test_data_new = test_data_new.rename(
            columns={'index': 0, 'Time': 'ts', 'UserId': 'user', 'ItemId': 'item', 'SessionId': 'session_id'})

        test_data_new['item'] = test_data_new['item'].apply(lambda x: self.item2id[x])
        test_data_new['user'] = test_data_new['user'].apply(lambda x: self.user2id[x])
        self.test_data = test_data_new
        self.current_session_id_test = -1
        self.current_pos_test = -1


    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=None):

        user_id = self.user2id[input_user_id]
        if session_id != self.current_session_id_test:
            self.current_session_id_test = session_id
            self.current_pos_test = 1
            self.current_session_test = self.test_data.loc[self.test_data['session_id'] == session_id]
            self.list_items = self.current_session_test['item'].values.tolist()


        self.feed_test(user_id, self.list_items[:self.current_pos_test + 1])

        test_data, test_loss, test_index, test_iterator, t_items = self.load_test()

        self.sess.run([tf.local_variables_initializer()])
        self.sess.run([test_iterator.initializer])

        # index è la tabella delle probabilità con (valori,indice)
        index, items = self.sess.run(
            [test_index, t_items])

        rmdir(self.test_path)

        self.current_pos_test +=1

        prob = pd.DataFrame(data=index.values.tolist(), index=list(self.item2id.keys()))[0]

        return prob

    def train_data_from_uid(self, user_id):
        user_data = self.train_data.loc[self.train_data['user'] == user_id]
        return user_data

    def feed_test(self, user_id, items):

        count = 1
        mkdir(self.test_path)
        writer = tf.io.TFRecordWriter(self.test_path + 'test.tfrecord')

        user_data = self.train_data_from_uid(user_id)

        all_sess = user_data['session_id'].unique()

        all_seq = items


        i = len(all_sess)

        sub_sess = [user_data[user_data['session_id'] == sess]['item'].values.tolist() for sess in all_sess[max(0,i-self.opt.max_session):i]]
        sub_node = np.hstack(sub_sess)
        for j in range(len(all_seq) - 1):
            features = {}
            sub_seq = all_seq[0:j + 1]
            features['tar'] = _int64_feature([all_seq[j + 1]])
            features['user'] = _int64_feature([user_id])
            node = np.unique(np.hstack([sub_node, sub_seq, [0]]))
            #生成每个session的别名和mask值，并且padding
            sub_sess_pad = [sess + [0]*(self.max_length-len(sess)) for sess in sub_sess]+[[0]*self.max_length]*(self.opt.max_session-len(sub_sess))
            sub_sess_alias = np.array([[np.where(node==s)[0][0] for s in sess_pad] for sess_pad in sub_sess_pad])
            features['session_alias'] = _int64_feature(sub_sess_alias.reshape(-1))
            features['session_alias_shape'] = _int64_feature(sub_sess_alias.shape)
            #session mask值
            features['session_mask'] =_int64_feature([len(sess) for sess in sub_sess]+[1]*(self.opt.max_session-len(sub_sess)))
            #session_len每个session序列中session的数量
            features['session_len'] = _int64_feature([len(sub_sess)])
            #生成seq别名和mask值并且padding
            sub_seq_pad = sub_seq#+[0]*(self.max_length-len(sub_seq))
            sub_seq_alias = [np.where(node == s)[0][0] for s in sub_seq_pad]
            features['seq_alias'] = _int64_feature(sub_seq_alias)
            #seq_pad.append(sub_seq_pad)
            #seq mask值
            features['seq_mask'] = _int64_feature([len(sub_seq)])
            #节点数量
            features['num_node'] = _int64_feature([len(node)])
            features['all_node'] = _int64_feature(node)
            if self.opt.graph == 'ggnn':
                u_A = np.zeros((len(node), len(node)))
            elif self.opt.graph == 'gcn':
                u_A = np.eye(len(node))
            for u_input in sub_sess:
                for k in np.arange(len(u_input)-1):
                    u = np.where(node == u_input[k])[0][0]
                    v = np.where(node == u_input[k + 1])[0][0]
                    if self.opt.adj == 'adj_all':
                        u_A[u][v] += 1
                    else:
                        u_A[u][v] = 1
            for l in np.arange(len(sub_seq)-1):
                u = np.where(node == sub_seq[l])[0][0]
                v = np.where(node == sub_seq[l + 1])[0][0]
                if self.opt.adj == 'adj_all':
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
#            if count%200 == 0 and i != end-1:
#                writer.close()
#                writer = tf.python_io.TFRecordWriter(orgin_path + str(count) + '.tfrecord')
        writer.close()


    def load_test(self):

        test_filenames = tf.train.match_filenames_once(self.test_path + 'test.tfrecord')

        test_dataset = tf.data.TFRecordDataset(test_filenames)

        test_dataset = test_dataset.map(parse_function_(self.opt.max_session))

        test_batch_padding_dataset = test_dataset.padded_batch(self.opt.batchSize, padded_shapes=self.padded_shape,
                                                               drop_remainder=True)
        test_iterator = test_batch_padding_dataset.make_initializable_iterator()

        test_data = test_iterator.get_next()

        with tf.variable_scope('model', reuse=True):
            test_loss, test_index, t_items = self.trainer.model.forward_test(test_data['A_in'], test_data['A_out'], test_data['all_node'],
                                                  test_data['seq_alias'], test_data['seq_mask'],
                                                  test_data['session_alias'],
                                                  test_data['session_len'], test_data['session_mask'], test_data['tar'],
                                                  test_data['user'], self.num_items)


        return test_data, test_loss, test_index, test_iterator, t_items


    def support_users(self):
        return True


    def predict_with_training_data(self):
        return False

    def clear(self):
        self.current_session = -1
        self.sess.close()
        tf.reset_default_graph()
        pass

    # 定义整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 定义浮点列表型的属性
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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

def rmdir(path):
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path, ignore_errors=True)
        return
    else:
        return

def apply_parallel(df_grouped, func):
    # TODO: insert number parallel job as parameters, 1 for debug
    Parallel(n_jobs=8)(delayed(func)(group) for name, group in df_grouped)

def generate_tfrecord(all_data, config, graph='ggnn', max_session=50, max_length=20, adj='adj'):




    def select_data(train=True):
        def select(data):
            orgin_path = config['train_path'] + '/user_' + str(np.unique(data[config['user_key']])[0]) + '/'
            mkdir(orgin_path)
            all_sess = data[config['session_key']].unique()

            user_id = data[config['user_key']].unique()[0]
            count = 1
            if len(all_sess) == 1:
                return None
            orgin_path = orgin_path + 'train_'
            start = 1
            end = len(all_sess)

            writer = tf.io.TFRecordWriter(orgin_path + str(count) + '.tfrecord')
            for i in range(start, end):
                # 生成session和seq
                if i < max_session + 1:
                    all_seq = data[data[config['session_key']] == all_sess[i]][config['item_key']].values.tolist()
                    sub_sess = [data[data[config['session_key']] == sess][config['item_key']].values.tolist() for sess in all_sess[0:i]]
                else:
                    all_seq = data[data[config['session_key']] == all_sess[i]][config['item_key']].values.tolist()
                    sub_sess = [data[data[config['session_key']] == sess][config['item_key']].values.tolist() for sess in
                                all_sess[i - max_session:i]]
                sub_node = np.hstack(sub_sess)
                for j in range(len(all_seq) - 1):
                    features = {}
                    sub_seq = all_seq[0:j + 1]
                    features['tar'] = _int64_feature([all_seq[j + 1]])
                    features[config['user_key']] = _int64_feature([user_id])
                    node = np.unique(np.hstack([sub_node, sub_seq, [0]]))
                    # 生成每个session的别名和mask值，并且padding
                    sub_sess_pad = [sess + [0] * (max_length - len(sess)) for sess in sub_sess] + [
                        [0] * max_length] * (max_session - len(sub_sess))
                    sub_sess_alias = np.array(
                        [[np.where(node == s)[0][0] for s in sess_pad] for sess_pad in sub_sess_pad])

                    features['session_alias'] = _int64_feature(sub_sess_alias.reshape(-1))

                    features['session_alias_shape'] = _int64_feature(sub_sess_alias.shape)
                    # session mask值
                    features['session_mask'] = _int64_feature(
                        [len(sess) for sess in sub_sess] + [1] * (max_session - len(sub_sess)))
                    # session_len每个session序列中session的数量
                    features['session_len'] = _int64_feature([len(sub_sess)])
                    # 生成seq别名和mask值并且padding
                    sub_seq_pad = sub_seq  # +[0]*(max_length-len(sub_seq))
                    sub_seq_alias = [np.where(node == s)[0][0] for s in sub_seq_pad]
                    features['seq_alias'] = _int64_feature(sub_seq_alias)
                    # seq_pad.append(sub_seq_pad)
                    # seq mask值
                    features['seq_mask'] = _int64_feature([len(sub_seq)])
                    # 节点数量
                    features['num_node'] = _int64_feature([len(node)])
                    features['all_node'] = _int64_feature(node)
                    if graph == 'ggnn':
                        u_A = np.zeros((len(node), len(node)))
                    elif graph == 'gcn':
                        u_A = np.eye(len(node))
                    for u_input in sub_sess:
                        for k in np.arange(len(u_input) - 1):
                            u = np.where(node == u_input[k])[0][0]
                            v = np.where(node == u_input[k + 1])[0][0]
                            if adj == 'adj_all':
                                u_A[u][v] += 1
                            else:
                                u_A[u][v] = 1
                    for l in np.arange(len(sub_seq) - 1):
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
                    # ------------稀疏方式------------
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
                    # --------------------------------------
                    tf_features = tf.train.Features(feature=features)
                    tf_example = tf.train.Example(features=tf_features)
                    tf_serialized = tf_example.SerializeToString()
                    writer.write(tf_serialized)
                    count += 1
                    if count % 200 == 0 and i != end - 1:
                        writer.close()
                        writer = tf.python_io.TFRecordWriter(orgin_path + str(count) + '.tfrecord')
            writer.close()
        return select

    apply_parallel(all_data.groupby(config['user_key']), select_data())


class Opt:
    def __init__(self, graph = 'ggnn', adj='adj',
                 max_session=100,
                 buffer_size=10000,
                 ggnn_drop=0.0,
                 epoch=30,
                 batchSize=50,
                 hiddenSize=100,
                 userSize=50,
                 decay=None,
                 l2=0.0,
                 lr=0.001,
                 mode='transformer',
                 decoder_attention=False,
                 encoder_attention=False,
                 user_=False,
                 history_=False,
                 behaviour_=False,
                 pool='max',
                 step=1):
        self.graph = graph
        self.adj = adj

        self.max_session = max_session
        self.buffer_size = buffer_size
        self.ggnn_drop = ggnn_drop
        self.epoch = epoch
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.userSize = userSize
        self.decay = decay
        self.l2 = l2
        self.lr = lr
        self.graph = graph
        self.mode = mode
        self.decoder_attention = decoder_attention
        self.encoder_attention = encoder_attention
        self.user_ = user_
        self.history_ = history_
        self.behaviour_ = behaviour_
        self.pool = pool
        self.step = step

        #todo: da eliminare data
        self.data = ''

    def set_max_length(self,max_length):
        self.max_length = max_length