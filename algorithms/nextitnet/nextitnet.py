import tensorflow as tf
import algorithms.nextitnet.data_loader_adapted as data_loader
import algorithms.nextitnet.generator_recsys as generator
import time
import pandas as pd
import numpy as np
from copy import copy
import traceback


class NextItNet:
    '''
    Code based on work by Yuan et al., A Simple but Hard-to-Beat Baseline for Session-based Recommendations, CoRR abs/1808.05163, 2018.

    # Strongly suggest running codes on GPU with more than 10G memory!!!
    # if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
    '''

    def __init__(self, test_path="", top_k=20, beta1=0.99, eval_iter=250, save_para_every=10000, kernel_size=3,
                 learning_rate=0.01, batch_size=128, iterations=10, dilations=[1,4,], dilated_channels=100, is_negsample=False, sampling_rate=0.2, limit_input_length=None, session_key='SessionId',
                 item_key='ItemId', time_key='Time'):

        '''
        :param top_k: Sample from top k predictions
        :param beta1: hyperpara-Adam
        :param eval_iter: Sample generator output evry x steps
        :param save_para_every: save model parameters every
        :param is_negsample:False #False denotes no negative sampling

        '''

        self.top_k = top_k
        self.beta1 = beta1
        self.eval_iter = eval_iter
        self.save_para_every = save_para_every
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iterations = iterations
        self.dilations = dilations
        self.dilated_channels = dilated_channels
        self.is_negsample = is_negsample
        self.sampling_rate = sampling_rate
        self.limit_input_length = limit_input_length
        self.test_path = test_path
        self.item_key = item_key
        self.time_key = time_key

        self.old_session_id = 0
        self.index_test = -1
        self.session = -1
        self.s = 0
        self.session_items = []
        self.session_key = session_key

    def fit(self, data, testdata):

        dl = data_loader.Data_Loader({'model_type': 'generator', 'limit_input_length': self.limit_input_length },
                                     data, testdata, self.session_key, self.item_key, self.time_key, pad_test=True)
        self.test_set = dl.item_test
        # all_samples = dl.item
        train_set = dl.item
        self.items = dl.item_dict
        self.mappingitem2idx = copy(self.items)
        self.itemrev = dl.reverse_dict
        self.max_session_length = dl.max_session_length
        self.input_limit = dl.input_limit
        # self.mappingitem2idx.pop("<UNK>")

        # Randomly shuffle data
        '''np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
        all_samples = all_samples[shuffle_indices]'''

        # Split train/test set
        '''dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
        train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]
'''

        # train_set = self.generatesubsequence( train_set )

        model_para = {
            # if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
            'item_size': len(self.items),
            'dilated_channels': self.dilated_channels,
            # if you use nextitnet_residual_block, you can use [1, 4, ],
            # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
            # when you change it do not forget to change it in nextitrec_generate.py
            # if you find removing residual network, the performance does not obviously decrease, then I think your data does not have strong seqeunce. Change a dataset and try again.
            'dilations': self.dilations,
            'kernel_size': self.kernel_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'iterations': self.iterations,
            'is_negsample': self.is_negsample, # False denotes no negative sampling
            'sampling_rate': self.sampling_rate
        }

        with tf.Graph().as_default():

            self.itemrec = generator.NextItNet_Decoder(model_para)
            self.itemrec.train_graph(model_para['is_negsample'])
            optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=self.beta1).minimize(
                self.itemrec.loss)
            self.itemrec.predict_graph(model_para['is_negsample'], reuse=True)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            saver = tf.train.Saver()

            numIters = 1
            tstart = time.time()

            for iter in range(model_para['iterations']):

                shuffle_train = np.random.permutation(np.arange(len(train_set)))
                train_set = train_set[shuffle_train]

                batch_no = 0
                batch_size = model_para['batch_size']
                while (batch_no + 1) * batch_size < train_set.shape[0]:

                    start = time.clock()

                    item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]

                    _, loss, results = self.sess.run(
                        [optimizer, self.itemrec.loss,
                         self.itemrec.arg_max_prediction],
                        feed_dict={
                            self.itemrec.itemseq_input: item_batch
                        })
                    end = time.clock()
                    if numIters % self.eval_iter == 0:
                        print("-------------------------------------------------------train1")
                        print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                            loss, iter, batch_no, numIters, train_set.shape[0] / batch_size))
                        print("TIME FOR BATCH", end - start)
                        print("TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0)
                    '''
                    if numIters % self.eval_iter == 0:
                        print("-------------------------------------------------------test1")
                        if (batch_no + 1) * batch_size < valid_set.shape[0]:
                            item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                        loss = sess.run(
                            [itemrec.loss_test],
                            feed_dict={
                                itemrec.input_predict: item_batch
                            })
                        print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                            loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size))'''

                    batch_no += 1
                    numIters += 1
                    if numIters % self.save_para_every == 0:
                        save_path = saver.save(self.sess,
                                               "algorithms/nextitnet/model_nextitnet.ckpt".format(iter, numIters))

                    if batch_no % 10 is 0:
                        print('train e{}: {} of {} batches in {}s'.format( iter, batch_no, len(train_set) / batch_size, time.time() - tstart ))

        index_list = [self.itemrev[a] for a in range(len(self.items))]
        self.index_list = list(map(lambda x: int(x) if x != '<UNK>' else -1, index_list))

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        if self.limit_input_length is None or self.limit_input_length is False:
            return self.predict_next_org(session_id, input_item_id, predict_for_item_ids, skip, type, timestamp)
        else:

            if self.session != session_id:  # new session
                self.session_items = []
                self.session = session_id
            if type == 'view':
                self.session_items.append(input_item_id)

            if skip:
                return

            input = [ self.items[i] for i in self.session_items[ (-self.input_limit-1) : ] ]
            input = ( max( self.input_limit - 1 - len( input ), 0 ) ) * [1] + input + [1]
            input = np.array( [input] )

            probs = self.sess.run(
                [self.itemrec.g_probs],
                feed_dict={
                    self.itemrec.input_predict: input
                })

            idx = self.input_limit-2
            res = pd.DataFrame(data=np.asanyarray(probs[0][0][idx].reshape(len(probs[0][0][idx]), 1)),
                             index=self.index_list)[0]

            return res


    def predict_next_org(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        # if numIters % args.eval_iter == 0:
        batch_size_test = 1
        if self.old_session_id != session_id:
            self.index_test += 1
            self.s = 0

            item_batch = self.test_set[self.index_test * batch_size_test:(self.index_test + 1) * batch_size_test, :]

            probs = self.sess.run(
                [self.itemrec.g_probs],
                feed_dict={
                    self.itemrec.input_predict: item_batch
                })

            self.probs = probs

            id0 = self.mappingitem2idx['0']
            self.current_sesslen = self.max_session_length - sum(item_batch[0] == id0)

        else:
            self.s = self.s + 1

        self.old_session_id = session_id

        if skip:
            return

        idx = self.max_session_length - self.current_sesslen + self.s
        g = pd.DataFrame(data=np.asanyarray( self.probs[0][0][idx].reshape(len(self.probs[0][0][idx]), 1)), index=self.index_list)[0]

        return g

    def generatesubsequence(self, train_set):
        # create subsession only for training
        subseqtrain = []
        for i in range(len(train_set)):
            # print x_train[i]
            seq = train_set[i]
            lenseq = len(seq)
            # session lens=100 shortest subsession=5 realvalue+95 0
            for j in range(lenseq - 2):
                subseqend = seq[:len(seq) - j]
                subseqbeg = [0] * j
                subseq = np.append(subseqbeg, subseqend)
                # beginseq=padzero+subseq
                # newsubseq=pad+subseq
                subseqtrain.append(subseq)
        x_train = np.array(subseqtrain)  # list to ndarray
        del subseqtrain
        # Randomly shuffle data
        np.random.seed(10)
        shuffle_train = np.random.permutation(np.arange(len(x_train)))
        x_train = x_train[shuffle_train]
        print("generating subsessions is done!")
        return x_train

    def clear(self):
        self.sess.close()
