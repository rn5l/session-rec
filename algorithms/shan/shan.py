# author：hucheng

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
import datetime

#shan_1: shan with 1+ session per user
class SHAN():
    def __init__(self, iter=100, global_dimension=50, neg_number=10, lambda_a = 0.01, lambda_uv = 0.01,
                 session_key='SessionId', item_key='ItemId', user_key ='UserId', time_key='Time'):
        self.lambda_u_v = lambda_uv
        self.lambda_a = lambda_a
        self.iter = iter
        self.global_dimensions = global_dimension
        self.neg_number = neg_number
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key

    def fit(self, train_data, test_data=None):

        data_key = dict()
        data_key['item_key'] = self.item_key
        data_key['time_key'] = self.time_key
        data_key['user_key'] = self.user_key
        data_key['session_key'] = self.session_key

        self.model = shan(train_data, test_data, self.neg_number, self.iter, self.global_dimensions, self.lambda_a, self.lambda_u_v ,data_key)
        self.model.build_model()
        self.model.run()
        pass
# 
    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=None):

        top_index_all, top_value_all = self.model.predict_next(session_id,input_item_id,input_user_id)

        pr = np.asanyarray(top_value_all)
        prob = pd.DataFrame(data=pr.T, index=list(top_index_all))[0]
        #if predict_for_item_ids is not None:
        #    prob = prob.loc[prob.index.isin(predict_for_item_ids)]
        return prob


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

    def clear(self):
        self.model.sess.close()
        tf.reset_default_graph()
        pass


class data_generation():
    def __init__(self, train_data, test_data, neg_number, data_key):
        print('init')

        self.item_key = data_key['item_key']
        self.time_key = data_key['time_key']
        self.user_key = data_key['user_key']
        self.session_key = data_key['session_key']

        self.train_users = []
        self.train_sessions = []  # 当前的session
        self.train_items = []  # 随机采样得到的positive
        self.train_neg_items = []  # 随机采样得到的negative
        self.train_pre_sessions = []  # 之前的session集合

        self.neg_number = neg_number
        self.user_number = 0
        self.item_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0
        self.load_data(train_data, test_data)

    def load_data(self, train_data, test_data):

        self.user_purchased_item = dict()

        self.max_length = train_data.groupby(['SessionId'])['ItemId'].count().max()

        self.itemids = train_data[self.item_key].unique()
        print("train data items: " + str(len(self.itemids)))
        assert(len(np.setdiff1d(test_data[self.item_key].unique(),self.itemids, assume_unique=True)) == 0)
        self.userids = np.union1d(train_data[self.user_key].unique(), test_data[self.user_key].unique())
        assert(len(np.setdiff1d(test_data[self.user_key].unique(), self.userids, assume_unique=True)) == 0)
        self.sessionids = np.union1d(train_data[self.session_key].unique(), test_data[self.session_key].unique())

        self.user_number = len(self.userids)
        self.item_number = len(self.itemids)
        self.item2id = dict(zip(self.itemids, range(0,len(self.itemids))))
        self.user2id = dict(zip(self.userids, range(0,len(self.userids))))
        self.session2id = dict(zip(self.sessionids, range(0, len(self.sessionids))))

        self.id2item = dict()
        for k in self.item2id.keys():
            self.id2item[self.item2id[k]] = k
        for user_id in train_data.UserId.unique():
            uid = self.user2id[user_id]
            user_pd = train_data.loc[train_data['UserId'] == user_id]

            sessions = []
            # warning can be ignored: checked with debug and user_pd change
            user_pd.loc[:,'ItemId'] = [self.item2id[iid] for iid in user_pd['ItemId']]
            for sess in user_pd.SessionId.unique():
                sessions.append(list(user_pd.loc[user_pd['SessionId'] == sess]['ItemId']))
            size = len(sessions)
            # this algo admit user with 1 session
            #if size < 2:
            #    continue
            the_first_session = sessions[0]
            self.train_pre_sessions.append(the_first_session)
            tmp = copy.deepcopy(the_first_session)
            self.user_purchased_item[uid] = tmp

            for j in range(1, size):
                    # 每个用户的每个session在train_users中都对应着其uid，不一定是连续的
                self.train_users.append(uid)
                # test = sessions[j].split(':')
                current_session = sessions[j]
                neg = self.gen_neg(uid)
                self.train_neg_items.append(neg)
                # 将当前session加入到用户购买的记录当中
                # 之所以放在这个位置，是因为在选择测试item时，需要将session中的一个item移除、
                # 如果放在后面操作，当前session中其实是少了一个用来做当前session进行预测的item
                if j != 1:
                    tmp = copy.deepcopy(self.user_purchased_item[uid])
                    self.train_pre_sessions.append(tmp)
                tmp = copy.deepcopy(current_session)
                self.user_purchased_item[uid].extend(tmp)
                # 随机挑选一个作为prediction item
                item = random.choice(current_session)
                self.train_items.append(item)
                current_session.remove(item)
                self.train_sessions.append(current_session)
                self.records_number += 1

        self.data = train_data

        self.test_candidate_items = list(range(self.item_number))
        self.test_sid_to_data = dict()
        for sess_id in test_data.SessionId.unique():
            sid = self.session2id[sess_id]
            session_pd = test_data.loc[test_data['SessionId'] == sess_id]

            session = [self.item2id[int(row)] for row in
                       session_pd['ItemId'].values]
            self.test_sid_to_data[sid] = session


    def get_test_data(self, sid, uid):

        return self.test_candidate_items, self.test_sid_to_data[sid], self.user_purchased_item[uid]


    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index

    def gen_neg(self, user_id):
        count = 0
        neg_item_set = list()
        while count < self.neg_number:
            neg_item = np.random.randint(self.item_number)
            if neg_item not in self.user_purchased_item[user_id]:
                neg_item_set.append(neg_item)
                count += 1
        return neg_item_set

    def gen_train_batch_data(self, batch_size):
        # l = len(self.train_users)

        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0

        batch_user = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        batch_item = self.train_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_session = self.train_sessions[self.train_batch_id]
        # batch_neg_item = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_neg_item = self.train_neg_items[self.train_batch_id]
        batch_pre_session = self.train_pre_sessions[self.train_batch_id]

        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session


class shan():

    def __init__(self, train_data, test_data, neg_number, itera, global_dimension, lambda_a, lambda_uv, data_key):
        print('init ... ')
        self.train_data = train_data
        self.test_data = test_data

        self.dg = data_generation(self.train_data, self.test_data, neg_number, data_key)
        # 数据格式化

        self.train_user_purchased_item_dict = self.dg.user_purchased_item

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number
        self.neg_number = self.dg.neg_number


        self.global_dimension = global_dimension
        self.batch_size = 1
        self.results = []  # 可用来保存test每个用户的预测结果，最终计算precision

        self.step = 0
        self.iteration = itera
        self.lamada_u_v = lambda_uv
        self.lamada_a = lambda_a

        self.current_session_test = -1
        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        # 不管是当前的session，还是之前的session集合，在数据处理阶段都是一个数组，数组内容为item的编号
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.pre_sessions = tf.placeholder(tf.int32, shape=[None], name='pre_sessions')
        self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        self.the_first_w = tf.get_variable('the_first_w', initializer=self.initializer_param,
                                           shape=[self.global_dimension, self.global_dimension])
        self.the_second_w = tf.get_variable('the_second_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])
        self.the_first_bias = tf.get_variable('the_first_bias', initializer=self.initializer_param,
                                              shape=[self.global_dimension])
        self.the_second_bias = tf.get_variable('the_second_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])

    def attention_level_one(self, user_embedding, pre_sessions_embedding, the_first_w, the_first_bias):

        # 由于维度的原因，matmul和multiply方法要维度的变化
        # 最终weight为 1*n 的矩阵
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(tf.nn.relu(
            tf.add(tf.matmul(pre_sessions_embedding, the_first_w), the_first_bias)), tf.transpose(user_embedding))))

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(self.weight)), axis=0)
        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding, the_second_w,
                            the_second_bias):
        # 需要将long_user_embedding加入到current_session_embedding中来进行attention，
        # 论文中规定，long_user_embedding的表示也不会根据softmax计算得到的参数而变化。

        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.relu(tf.add(
                tf.matmul(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                          the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def build_model(self):
        print('building model ... ')
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)
        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)

        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.pre_sessions_embedding,
                                                            self.the_first_w, self.the_first_bias)

        self.hybrid_user_embedding = self.attention_level_two(self.user_embedding, self.long_user_embedding,
                                                              self.current_session_embedding,
                                                              self.the_second_w, self.the_second_bias)

        # compute preference
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding))
        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding))
        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        self.regular_loss_u_v = tf.add(tf.add(self.lamada_u_v * tf.nn.l2_loss(self.user_embedding),
                                              self.lamada_u_v * tf.nn.l2_loss(self.item_embedding)),
                                       self.lamada_u_v * tf.nn.l2_loss(self.neg_item_embedding))
        self.regular_loss_a = tf.add(self.lamada_a * tf.nn.l2_loss(self.the_first_w),
                                     self.lamada_a * tf.nn.l2_loss(self.the_second_w))
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)

        # 增加test操作，由于每个用户pre_sessions和current_session的长度不一样，
        # 所以无法使用同一个矩阵进行表示同时计算，因此每个user计算一次，将结果保留并进行统计
        # 注意，test集合的整个item_embeeding得到的是 [M*K]的矩阵，M为所有item的个数，K为维度
        self.top_value_10, self.top_index_10 = tf.nn.top_k(self.positive_element_wise, k=10, sorted=True)
        self.top_value_20, self.top_index_20 = tf.nn.top_k(self.positive_element_wise, k=20, sorted=True)
        self.top_value_50, self.top_index_50 = tf.nn.top_k(self.positive_element_wise, k=50, sorted=True)
        self.top_value_all, self.top_index_all = tf.nn.top_k(self.positive_element_wise, k=self.item_number, sorted=False)

    def run(self):
        print('running ... ')

        self.sess = tf.Session()

        self.intention_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(
            self.intention_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for iter in range(self.iteration):
            print(datetime.datetime.now())
            print('new iteration begin ... ')
            print('iteration: '+str(iter))

            all_loss = 0
            while self.step * self.batch_size < self.dg.records_number:
                # 按批次读取数据
                batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions = self.dg.gen_train_batch_data(
                    self.batch_size)

                _, loss = self.sess.run([self.intention_optimizer, self.intention_loss],
                                        feed_dict={self.user_id: batch_user,
                                                   self.item_id: batch_item,
                                                   self.current_session: batch_session,
                                                   self.neg_item_id: batch_neg_item,
                                                   self.pre_sessions: batch_pre_sessions
                                                   })
                all_loss += loss
                self.step += 1
            print('loss = '+str(all_loss)+'\n')
            print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
            self.step = 0


    def predict_next(self, session_id, item_id, user_id):

        if self.current_session_test != session_id:
            self.current_session_test = session_id
            self.batch_item, \
            self.batch_session, \
            self.batch_pre_session = self.dg.get_test_data(self.dg.session2id[session_id]
                                                           ,self.dg.user2id[user_id])
            self.current_pos = 0
        else:
            self.current_pos += 1

        top_index_all, top_value_all = self.sess.run(
                [self.top_index_all, self.top_value_all],
                feed_dict={self.user_id: [self.dg.user2id[user_id]],
                           self.item_id: self.batch_item,
                           self.current_session: self.batch_session[:self.current_pos],
                           self.pre_sessions: self.batch_pre_session})

        index_all = [self.dg.id2item[x] for x in top_index_all.tolist()[0]]

        return index_all, top_value_all

