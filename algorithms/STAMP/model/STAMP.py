# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from algorithms.STAMP.basic_layer.NN_adam import NN
from algorithms.STAMP.util.Printer import TIPrint
from algorithms.STAMP.util.batcher.equal_len.batcher_p import batcher
from algorithms.STAMP.util.AccCalculater import cau_recall_mrr_org
from algorithms.STAMP.util.AccCalculater import cau_samples_recall_mrr
from algorithms.STAMP.util.Pooler import pooler
from algorithms.STAMP.basic_layer.FwNn3AttLayer import FwNnAttLayer
from algorithms.STAMP.data_prepare.load_dict import load_random
from algorithms.STAMP.data_prepare.dataset_read import load_data
from algorithms.STAMP.util.Config import read_conf
from algorithms.STAMP.util.FileDumpLoad import dump_file, load_file
from algorithms.STAMP.util.Randomer import Randomer
from copy import copy

mid_rsc15_train_data = "rsc15_train.data"
mid_rsc15_test_data = "rsc15_test.data"
mid_rsc15_emb_dict = "rsc15_emb_dict.data"
mid_rsc15_4_emb_dict = "rsc15_4_emb_dict.data"
mid_rsc15_64_emb_dict = "rsc15_64_emb_dict.data"
mid_cikm16_emb_dict = "cikm16_emb_dict.data"


class Seq2SeqAttNN(NN):
    """
    Code based on work by Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD 2018.

    The memory network with context attention.
    """

    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, model="stamp", dataset="rsc15_64", recsys_threshold_acc=0.68, cikm_threshold_acc=0.62,
                 n_epochs=10, hidden_size=100, active="sigmoid", decay_steps=1, decay_rate=0.1, batch_size=512,
                 init_lr=0.003, stddev=0.05, emb_stddev=0.002, edim=100, max_grad_norm=110, pad_idx=0, emb_up=True,
                 update_lr=False, model_save_path="algorithms/STAMP/saved_models/", is_print=True, cell="gru",
                 cut_off=20, reload=True, class_num=3, is_train=False, is_save=False,
                 model_path="algorithms/STAMP/saved_models/", session_key='SessionId', item_key='ItemId',
                 time_key='Time'):
        # self.rsc15_train=rsc15_train
        # self.rsc15_test = rsc15_test
        # self.cikm16_train=cikm16_train
        # self.cikm16_test=cikm16_test
        self.recsys_threshold_acc = recsys_threshold_acc
        self.cikm_threshold_acc = cikm_threshold_acc
        # self.module, self.obj, config, self.options = self.load_conf(model, modelconf)
        self.emb_stddev = emb_stddev
        self.hidden_size = hidden_size
        self.datas = dataset
        self.reload = reload
        # self.train_data, self.test_data, self.item2idx = self.load_tt_datas(self.reload)
        # self.mappingitem2idx=copy(self.item2idx)
        # self.mappingitem2idx.pop("<pad>")
        self.s = 0
        self.old_session_id = 0
        print(model)
        self.class_num = class_num
        self.nepoch = n_epochs
        config = {}
        config['init_lr'] = init_lr
        config['update_lr'] = update_lr
        config['max_grad_norm'] = max_grad_norm
        config['decay_steps'] = decay_steps
        config['decay_rate'] = decay_rate
        config['class_num'] = class_num
        config['dataset'] = dataset
        config['model_save_path'] = model_save_path
        config['model'] = model
        config['saved_model'] = model_path
        super(Seq2SeqAttNN, self).__init__(config)
        self.batch_size = batch_size  # the max train batch size.
        self.init_lr = init_lr  # the initialize learning rate.
        # the base of the initialization of the parameters.
        self.stddev = stddev
        self.edim = edim  # the dim of the embedding.
        self.max_grad_norm = max_grad_norm  # the L2 norm.
        # the pad id in the embedding dictionary.
        self.pad_idx = pad_idx

        # the pre-train embedding.
        ## shape = [nwords, edim]

        # update the pre-train embedding or not.
        self.emb_up = emb_up

        # the active function.
        self.active = active

        # hidden size
        self.hidden_size = hidden_size

        self.is_print = is_print
        self.is_save = is_save
        self.is_train = is_train
        self.cut_off = cut_off
        self.is_first = True
        # the input.
        self.inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

    def load_conf(self, model, modelconf):
        '''
        model: 需要加载的模型
        modelconf: model config文件所在的路径
        '''
        # load model config
        # print(str(modelconf))
        model_conf = read_conf(model, modelconf)
        if model_conf is None:
            raise Exception("wrong model config path.", model_conf)
        module = model_conf['module']
        obj = model_conf['object']
        params = model_conf['params']
        options = model_conf['options']
        params = params.split("/")
        paramconf = ""
        # last element of the list
        model = params[-1]
        for line in params[:-1]:
            paramconf += line + "/"
        # "config/nn_param.conf"
        paramconf = paramconf[:-1]
        # load super params.
        param_conf = read_conf(model, paramconf)

        options_conf = read_conf(model, options)
        return module, obj, param_conf, options_conf

    def load_tt_datas(self, train, test, reload=True):
        '''
        loda data.
        config: 获得需要加载的数据类型，放入pre_embedding.
        nload: 是否重新解析原始数据
        '''

        if reload:
            print("reload the datasets.")
            print(self.datas)

            train_data, test_data, item2idx, n_items = load_data(
                train,
                test,
                self.session_key,
                self.item_key,
                self.time_key
            )
            self.n_items = n_items - 1
            emb_dict = load_random(item2idx, edim=self.hidden_size, init_std=self.emb_stddev)
            self.pre_embedding = emb_dict
            #path = 'algorithms/STAMP/mid_data/mid_data_'
            #dump_file([emb_dict, path + self.datas])
            print("-----")

        else:
            print("not reload the datasets.")
            print(self.datas)

            train_data, test_data, item2idx, n_items = load_data(
                train,
                test,
            )
            self.n_items = n_items - 1
            path = 'algorithms/STAMP/mid_data/mid_data_'
            emb_dict = load_file(path + self.datas)
            self.pre_embedding = emb_dict[0]
            print("-----")
        return train_data, test_data, item2idx
        # #CHECK
        # if self.datas == 'rsc15':
        #     train_data, test_data, item2idx, n_items = load_data_p(
        #         self.rsc15_train,
        #         self.rsc15_test,
        #         pro=1
        #     )
        #
        #     self.n_items= n_items - 1
        #     emb_dict = load_random(item2idx, edim=self.hidden_size, init_std=self.emb_stddev)
        #     self.pre_embedding = emb_dict
        #     path = '../mid_data'
        #     dump_file([emb_dict, path + self.datas])
        #     print("-----")
        #
        # if self.datas == 'rsc15_4':
        #     train_data, test_data, item2idx, n_items = load_data_p(
        #         self.rsc15_train,
        #         self.rsc15_test,
        #         pro=4
        #     )
        #
        #     self.n_items = n_items - 1
        #     emb_dict = load_random(item2idx, edim=self.hidden_size, init_std=self.emb_stddev)
        #     self.pre_embedding = emb_dict
        #     path = '../mid_data'
        #     dump_file([emb_dict, path + mid_rsc15_4_emb_dict])
        #     print("-----")
        #
        # if self.datas== 'rsc15_64':
        #     train_data, test_data, item2idx, n_items = load_data_p(
        #         self.rsc15_train,
        #         self.rsc15_test,
        #         pro=64
        #     )
        #
        #     self.n_items = n_items - 1
        #     emb_dict = load_random(item2idx, edim=self.hidden_size, init_std=self.emb_stddev)
        #     self.pre_embedding = emb_dict
        #     path = '../mid_data'
        #     dump_file([emb_dict, path + mid_rsc15_64_emb_dict])
        #     print("-----")
        #
        # if self.datas == 'cikm16':
        #     train_data, test_data,item2idx, n_items = load_data2(
        #         self.cikm16_train,
        #         self.cikm16_test,
        #         class_num=self.class_num
        #     )
        #     self.n_items = n_items - 1
        #     emb_dict = load_random(item2idx, edim=self.hidden_size, init_std=self.emb_stddev)
        #     self.pre_embedding = emb_dict
        #     path = '../mid_data'
        #     dump_file([emb_dict, path + mid_cikm16_emb_dict])
        #     print("-----")

        # else:
        #     print("not reload the datasets.")
        #     print(self.datas)
        #
        #     if self.datas == 'rsc15':
        #         train_data, test_data, item2idx, n_items = load_data_p(
        #             self.rsc15_train,
        #             self.rsc15_test,
        #             pro=1
        #         )
        #
        #         self.n_items = n_items - 1
        #         path = '../mid_data'
        #         emb_dict = load_file(path + mid_rsc15_emb_dict)
        #         self.pre_embedding = emb_dict[0]
        #         print("-----")
        #
        #     if self.datas == 'rsc15_4':
        #         train_data, test_data, item2idx, n_items = load_data_p(
        #             self.rsc15_train,
        #             self.rsc15_test,
        #             pro=4
        #         )
        #
        #         self.n_items = n_items - 1
        #         path = '../mid_data'
        #         emb_dict = load_file(path + mid_rsc15_4_emb_dict)
        #         self.pre_embedding = emb_dict[0]
        #         print("-----")
        #
        #     if self.datas == 'rsc15_64':
        #         train_data, test_data, item2idx, n_items = load_data_p(
        #             self.rsc15_train,
        #             self.rsc15_test,
        #             pro=64
        #         )
        #
        #         self.n_items = n_items - 1
        #         path = '../mid_data'
        #         emb_dict = load_file(path + mid_rsc15_64_emb_dict)
        #         self.pre_embedding= emb_dict[0]
        #         print("-----")
        #
        #     if self.datas == 'cikm16':
        #         train_data, test_data, item2idx, n_items = load_data2(
        #             self.cikm16_train,
        #             self.cikm16_test,
        #             class_num=self.class_num
        #         )
        #         self.n_items = n_items - 1
        #         path = '../mid_data'
        #         emb_dict = load_file(path + mid_cikm16_emb_dict)
        #         self.pre_embedding = emb_dict[0]
        #         print("-----")

    def build_model(self):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None, None],
            name="inputs"
        )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )

        batch_size = tf.shape(self.inputs)[0]

        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )

        self.lab_input = tf.placeholder(
            tf.int32,
            [None],
            name="lab_input"
        )

        # the lookup dict.
        self.embe_dict = tf.Variable(
            self.pre_embedding,
            dtype=tf.float32,
            trainable=self.emb_up
        )

        self.pe_mask = tf.Variable(
            self.pre_embedding_mask,
            dtype=tf.float32,
            trainable=False
        )
        self.embe_dict *= self.pe_mask

        sent_bitmap = tf.ones_like(tf.cast(self.inputs, tf.float32))

        inputs = tf.nn.embedding_lookup(self.embe_dict, self.inputs, max_norm=1)
        lastinputs = tf.nn.embedding_lookup(self.embe_dict, self.last_inputs, max_norm=1)

        org_memory = inputs

        pool_out = pooler(
            org_memory,
            'mean',
            axis=1,
            sequence_length=tf.cast(tf.reshape(self.sequence_length, [batch_size, 1]), tf.float32)
        )
        pool_out = tf.reshape(pool_out, [-1, self.hidden_size])

        attlayer = FwNnAttLayer(
            self.edim,
            active=self.active,
            stddev=self.stddev,
            norm_type='none'
        )
        attout, alph = attlayer.forward(org_memory, lastinputs, pool_out, sent_bitmap)
        attout = tf.reshape(attout, [-1, self.edim]) + pool_out
        self.alph = tf.reshape(alph, [batch_size, 1, -1])

        self.w1 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )

        self.w2 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )
        attout = tf.tanh(tf.matmul(attout, self.w1))
        # attout = tf.nn.dropout(attout, self.output_keep_probs)
        lastinputs = tf.tanh(tf.matmul(lastinputs, self.w2))
        # lastinputs= tf.nn.dropout(lastinputs, self.output_keep_probs)
        prod = attout * lastinputs
        sco_mat = tf.matmul(prod, self.embe_dict[1:], transpose_b=True)
        self.softmax_input = sco_mat
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat, labels=self.lab_input)

        # the optimize.
        self.params = tf.trainable_variables()
        self.optimize = super(Seq2SeqAttNN, self).optimize_normal(
            self.loss, self.params)

    def fit(self, train, test):
        self.train_data, self.test_data, self.item2idx = self.load_tt_datas(train, test, self.reload)
        # generate the pre_embedding mask.
        self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
        self.pre_embedding_mask[self.pad_idx] = 0
        self.mappingitem2idx = copy(self.item2idx)
        self.mappingitem2idx.pop("<pad>")
        # module = __import__(self.module, fromlist=True)
        # setup randomer
        Randomer.set_stddev(self.stddev)

        with tf.Graph().as_default():
            # build model
            self.build_model()
            if self.is_save or not self.is_train:
                self.saver = tf.train.Saver(max_to_keep=30)
            else:
                self.saver = None
            # run

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            if self.datas == "cikm16":
                self.train(self.train_data, self.saver, threshold_acc=self.cikm_threshold_acc)
            else:
                self.train(self.train_data, self.saver, threshold_acc=self.recsys_threshold_acc)

    def train(self, train_data, saver=None, threshold_acc=0.99):
        for epoch in range(self.nepoch):  # epoch round.
            batch = 0
            c = []
            cost = 0.0  # the cost of each epoch.
            bt = batcher(
                samples=train_data.samples,
                class_num=self.n_items,
                random=True
            )
            
            start_time = time.time()
            
            while bt.has_next():  # batch round.
                # get this batch data
                batch_data = bt.next_batch()
                # build the feed_dict
                # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
                batch_lenth = len(batch_data['in_idxes'])
                event = len(batch_data['in_idxes'][0])

                if batch_lenth > self.batch_size:
                    patch_len = int(batch_lenth / self.batch_size)
                    remain = int(batch_lenth % self.batch_size)
                    i = 0
                    for x in range(patch_len):
                        tmp_in_data = batch_data['in_idxes'][i:i + self.batch_size]
                        tmp_out_data = batch_data['out_idxes'][i:i + self.batch_size]
                        for s in range(len(tmp_in_data[0])):
                            batch_in = []
                            batch_out = []
                            batch_last = []
                            batch_seq_l = []
                            for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                                _in = tmp_in[s]
                                _out = tmp_out[s] - 1
                                batch_last.append(_in)
                                batch_in.append(tmp_in[:s + 1])
                                batch_out.append(_out)
                                batch_seq_l.append(s + 1)
                            feed_dict = {
                                self.inputs: batch_in,
                                self.last_inputs: batch_last,
                                self.lab_input: batch_out,
                                self.sequence_length: batch_seq_l

                            }
                            # train
                            crt_loss, crt_step, opt, embe_dict = self.sess.run(
                                [self.loss, self.global_step, self.optimize, self.embe_dict],
                                feed_dict=feed_dict
                            )

                            # cost = np.mean(crt_loss)
                            c += list(crt_loss)
                            # print("Batch:" + str(batch) + ",cost:" + str(cost))
                            batch += 1
                        i += self.batch_size
                    if remain > 0:
                        # print (i, remain)
                        tmp_in_data = batch_data['in_idxes'][i:]
                        tmp_out_data = batch_data['out_idxes'][i:]
                        for s in range(len(tmp_in_data[0])):
                            batch_in = []
                            batch_out = []
                            batch_last = []
                            batch_seq_l = []
                            for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                                _in = tmp_in[s]
                                _out = tmp_out[s] - 1
                                batch_last.append(_in)
                                batch_in.append(tmp_in[:s + 1])
                                batch_out.append(_out)
                                batch_seq_l.append(s + 1)
                            feed_dict = {
                                self.inputs: batch_in,
                                self.last_inputs: batch_last,
                                self.lab_input: batch_out,
                                self.sequence_length: batch_seq_l

                            }
                            # train
                            crt_loss, crt_step, opt, embe_dict = self.sess.run(
                                [self.loss, self.global_step, self.optimize, self.embe_dict],
                                feed_dict=feed_dict
                            )

                            # cost = np.mean(crt_loss)
                            c += list(crt_loss)
                            # print("Batch:" + str(batch) + ",cost:" + str(cost))
                            batch += 1
                else:
                    tmp_in_data = batch_data['in_idxes']
                    tmp_out_data = batch_data['out_idxes']
                    for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            _in = tmp_in[s]
                            _out = tmp_out[s] - 1
                            batch_last.append(_in)
                            batch_in.append(tmp_in[:s + 1])
                            batch_out.append(_out)
                            batch_seq_l.append(s + 1)
                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l

                        }
                        # train
                        crt_loss, crt_step, opt, embe_dict = self.sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )

                        # cost = np.mean(crt_loss)
                        c += list(crt_loss)
                        # print("Batch:" + str(batch) + ",cost:" + str(cost))
                        batch += 1
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}\ttime: {}s'.format(epoch, avgc, (time.time()-start_time)))

        # SAVE MODEL
        # self.save_model(sess, self.config, saver)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):

        '''
        Gives prediction scores for a selected item in a selected session.
        The self.s variable allow to shift the items in the selected session.
        Parameters
        --------
        session_id : int
            Contains the session ID.
        input_item_id : int
            Contains the item ID of the events of the session.
        Returns
        --------
        out : pandas.DataFrame
            Prediction scores given the input_item_id and session_id for the next item.
            Columns: 1 column containing the scores; rows: items. Rows are indexed by the item IDs.'''

        sample = [x for x in self.test_data.samples if x.session_id == session_id]
        if self.old_session_id != session_id:
            self.s = 0

        if skip:
            self.s = self.s + 1
            self.old_session_id = session_id
            return

        c_loss = []
        bt = batcher(
            samples=sample,
            class_num=self.n_items,
            random=False
        )

        while bt.has_next():  # batch round.
            batch_data = bt.next_batch()

            tmp_in_data = batch_data['in_idxes']
            tmp_out_data = batch_data['out_idxes']
            tmp_batch_ids = batch_data['batch_ids']
            # for s in range(len(tmp_in_data[0])):
            batch_in = []
            batch_out = []
            batch_last = []
            batch_seq_l = []
            for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                _in = tmp_in[self.s]
                _out = tmp_out[self.s] - 1
                batch_last.append(_in)
                batch_in.append(tmp_in[:self.s + 1])
                batch_out.append(_out)
                batch_seq_l.append(self.s + 1)
            feed_dict = {
                self.inputs: batch_in,
                self.last_inputs: batch_last,
                self.lab_input: batch_out,
                self.sequence_length: batch_seq_l

            }

            preds, loss, alpha = self.sess.run(
                [self.softmax_input, self.loss, self.alph],
                feed_dict=feed_dict
            )

            # CHECK
            # t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
            # I DON'T KNOW
            self.test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
            c_loss += list(loss)
            self.s = self.s + 1
            self.old_session_id = session_id
            return \
            pd.DataFrame(data=np.asanyarray(preds.reshape(len(preds[0]), 1)), index=list(self.mappingitem2idx.keys()))[
                0]

    # def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch_size=100):
    #     '''
    #     Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.
    #
    #     If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
    #     If it changes, the hidden state of the network is reset to zeros.
    #
    #     Parameters
    #     --------
    #     session_ids : 1D array
    #         Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
    #     input_item_ids : 1D array
    #         Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
    #     predict_for_item_ids : 1D array (optional)
    #         IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
    #     batch : int
    #         Prediction batch size.
    #
    #     Returns
    #     --------
    #     out : pandas.DataFrame
    #         Prediction scores for selected items for every event of the batch.
    #         Columns: events of the batch; rows: items. Rows are indexed by the item IDs.'''
    #
    #     # calculate the acc
    #     sample = filter(lambda x: x.session_id == session_ids, self.test_data.samples)
    #     #print('Measuring Recall@{} and MRR@{}'.format(self.cut_off, self.cut_off))
    #     self.batch_size=batch_size
    #     mrr, recall = [], []
    #     c_loss =[]
    #     batch = 0
    #     bt = batcher(
    #         samples =sample,
    #         class_num = self.n_items,
    #         random = False
    #     )
    #     while bt.has_next():    # batch round.
    #         # get this batch data
    #         batch_data = sample
    #         # build the feed_dict
    #         # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
    #         # batch_lenth = len(batch_data['in_idxes'])
    #         # event = len(batch_data['in_idxes'][0])
    #         # if batch_lenth > self.batch_size:
    #         #     patch_len = int(batch_lenth / self.batch_size)
    #         #     remain = int(batch_lenth % self.batch_size)
    #         #     i = 0
    #         #     for x in range(patch_len):
    #         #         tmp_in_data = batch_data['in_idxes'][i:i+self.batch_size]
    #         #         tmp_out_data = batch_data['out_idxes'][i:i+self.batch_size]
    #         #         tmp_batch_ids = batch_data['batch_ids'][i:i+self.batch_size]
    #         #         for s in range(len(tmp_in_data[0])):
    #         #             batch_in = []
    #         #             batch_out = []
    #         #             batch_last = []
    #         #             batch_seq_l = []
    #         #             for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
    #         #                 _in = tmp_in[s]
    #         #                 _out = tmp_out[s] - 1
    #         #                 batch_last.append(_in)
    #         #                 batch_in.append(tmp_in[:s + 1])
    #         #                 batch_out.append(_out)
    #         #                 batch_seq_l.append(s + 1)
    #         #             feed_dict = {
    #         #                 self.inputs: batch_in,
    #         #                 self.last_inputs: batch_last,
    #         #                 self.lab_input: batch_out,
    #         #                 self.sequence_length: batch_seq_l
    #         #
    #         #             }
    #         #             # train
    #         #             preds, loss, alpha = self.sess.run(
    #         #                 [self.softmax_input, self.loss, self.alph],
    #         #                 feed_dict=feed_dict
    #         #             )
    #         #             #return pd.DataFrame(data=preds, index=predict_for_item_ids)
    #         #             t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
    #         #             self.test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
    #         #             self.test_data.pack_preds(ranks, tmp_batch_ids)
    #         #             c_loss += list(loss)
    #         #             #recall += t_r
    #         #             #mrr += t_m
    #         #             batch += 1
    #         #         i += self.batch_size
    #         #     if remain > 0:
    #         #         # print (i, remain)
    #         #         tmp_in_data = batch_data['in_idxes'][i:]
    #         #         tmp_out_data = batch_data['out_idxes'][i:]
    #         #         tmp_batch_ids = batch_data['batch_ids'][i:]
    #         #         for s in range(len(tmp_in_data[0])):
    #         #             batch_in = []
    #         #             batch_out = []
    #         #             batch_last = []
    #         #             batch_seq_l = []
    #         #             for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
    #         #                 _in = tmp_in[s]
    #         #                 _out = tmp_out[s] - 1
    #         #                 batch_last.append(_in)
    #         #                 batch_in.append(tmp_in[:s + 1])
    #         #                 batch_out.append(_out)
    #         #                 batch_seq_l.append(s + 1)
    #         #             feed_dict = {
    #         #                 self.inputs: batch_in,
    #         #                 self.last_inputs: batch_last,
    #         #                 self.lab_input: batch_out,
    #         #                 self.sequence_length: batch_seq_l
    #         #
    #         #             }
    #         #
    #         #             # train
    #         #             preds, loss, alpha = self.sess.run(
    #         #                 [self.softmax_input, self.loss, self.alph],
    #         #                 feed_dict=feed_dict
    #         #             )
    #         #             t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
    #         #             self.test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
    #         #             self.test_data.pack_preds(ranks, tmp_batch_ids)
    #         #             c_loss += list(loss)
    #         #             recall += t_r
    #         #             mrr += t_m
    #         #             batch += 1
    #         # else:
    #         tmp_in_data = batch_data['in_idxes']
    #         tmp_out_data = batch_data['out_idxes']
    #         tmp_batch_ids = batch_data['batch_ids']
    #         for s in range(len(tmp_in_data[0])):
    #             batch_in = []
    #             batch_out = []
    #             batch_last = []
    #             batch_seq_l = []
    #             for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
    #                 _in = tmp_in[s]
    #                 _out = tmp_out[s] - 1
    #                 batch_last.append(_in)
    #                 batch_in.append(tmp_in[:s + 1])
    #                 batch_out.append(_out)
    #                 batch_seq_l.append(s + 1)
    #             feed_dict = {
    #                 self.inputs: batch_in,
    #                 self.last_inputs: batch_last,
    #                 self.lab_input: batch_out,
    #                 self.sequence_length: batch_seq_l
    #
    #             }
    #             # train
    #             preds, loss, alpha = self.sess.run(
    #                 [self.softmax_input, self.loss, self.alph],
    #                 feed_dict=feed_dict
    #             )
    #
    #             #CHECK
    #
    #             #t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
    #             #I DON'T KNOW
    #             self.test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
    #             #self.test_data.pack_preds(ranks, tmp_batch_ids)
    #
    #             c_loss += list(loss)
    #             return pd.DataFrame(data=preds.reshape(len(preds),1), index=list(self.item2idx.keys()))
    #     #         recall += t_r
    #     #         mrr += t_m
    #     #         batch += 1
    #     # r, m =cau_samples_recall_mrr(self.test_data.samples,self.cut_off)
    #     # print (r,m)
    #     # print (np.mean(c_loss))
    #     # return  np.mean(recall), np.mean(mrr)

    def clear(self):
        self.sess.close()
        pass

