import os
import importlib
os.environ["CHAINER_TYPE_CHECK"] = "0"
import sys
importlib.reload(sys)
#sys.setdefaultencoding('utf-8')
from algorithms.RepeatNet.base.corpus import *
import pickle
import argparse
import random
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers,training,reporter,iterators
#from chainer.training.updaters import MultiprocessParallelUpdater
from chainer.iterators import MultiprocessIterator
import codecs
import collections
from algorithms.RepeatNet.gru_rec import *
from algorithms.RepeatNet.att_rec import *
from algorithms.RepeatNet.repeat_net import *
from algorithms.RepeatNet.base.recsys import *
import pandas as pd

print('Chainer Version: '+chainer.__version__)
print('Cupy Version: ', cuda.cupy.__version__)
#print('CuDNN Version: ',chainer._cudnn_version)


class RN:
    def __init__(self, embed_size=100, hidden_size=100, epochs=10, lr=0.001, train_batch_size=1024, session_key='SessionId', item_key='ItemId'):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.epochs = epochs
        self.lr = lr
        self.pointer = 0
        self.train_batch_size=train_batch_size
        self.session_key = session_key
        self.item_key = item_key


    def fit(self, data, test=None):
        with chainer.using_config('cudnn_deterministic', True):
            with chainer.using_config('use_cudnn', 'auto'):
                print(chainer.config.show())

                device =0
                seed = 42
                random.seed(seed)
                np.random.seed(seed)
                cuda.cupy.random.seed(seed)

                self.item2id, self.id2item = load_item(data,self.session_key,self.item_key)

                #print("start")
                model_name = 'repeat_rec_100_100_digi'
                print(model_name, 'GPU: ', device)
                train_batchsize = 1024
                train_dataset = SessionCorpus(data, item2id=self.item2id).load(self.session_key,self.item_key)
                print("train loaded")
                # valid_dataset = SessionCorpus(file_path='data/yoo_1_4/valid_1_over_4.txt', item2id=item2id).load()
                self.test_dataset = SessionCorpus(test, item2id=self.item2id).load(self.session_key,self.item_key)
                test_batchsize = 1024

                # model=AttRec(len(item2id), embed_size=100, hidden_size=100)
                # model = NoAttRepeatNet(len(item2id), embed_size=100, hidden_size=100, joint_train=False)
                model = RepeatNet(len(self.item2id), self.embed_size, self.hidden_size, joint_train=False)
                self.recsys = RecSys(model, len(self.item2id))
                # serializers.load_npz('model/repeat_rec_100_100_yoo.model.52012.npz', recsys)

                optimizer = optimizers.Adam(alpha=self.lr)
                # optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-6)
                optimizer.setup(self.recsys)
                optimizer.add_hook(chainer.optimizer.GradientClipping(5))

                # serializers.load_npz('model/repeat_rec_100_100_yoo.optimizer.52012.npz', optimizer)
                def change_alpha(trainer):
                    # if updater.epoch>10:
                    optimizer.alpha = optimizer.alpha * 0.5
                    print('change step size to ', optimizer.alpha)
                    return

                train_iter = iterators.SerialIterator(train_dataset, batch_size=train_batchsize)
                # train_iters = [
                #     chainer.iterators.MultiprocessIterator(train_batch_i, train_batchsize, shuffle=False)
                #     for train_batch_i in chainer.datasets.split_dataset_n_random(train_dataset, len(devices))]
                # valid_iter = iterators.SerialIterator(valid_dataset, batch_size=len(valid_dataset), shuffle=False, repeat=False)
                test_iter = iterators.SerialIterator(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False,
                                                     repeat=False)

                def converter(batch, device):
                    return batch

                updater = training.StandardUpdater(train_iter, optimizer, converter=converter, device=device)
                # updater = MultiprocessParallelUpdater(train_iters, optimizer, converter=converter, devices=devices)
                #trainer = training.Trainer(updater, (self.epochs, 'epoch'))
                trainer = training.Trainer(updater)

                trainer.out = 'model/'
                trainer.extend(training.extensions.LogReport(trigger=(100, 'iteration')))
                trainer.extend(training.extensions.PrintReport(
                    ['epoch', 'iteration', 'main/loss', 'main/loss0', 'main/loss1', 'main/loss2',
                     'validation/main/valid/mrr5',
                     'validation/main/valid/recall5',
                     'validation/main/valid/mrr10',
                     'validation/main/valid/recall10',
                     'validation/main/valid/mrr15',
                     'validation/main/valid/recall15',
                     'validation/main/valid/mrr20',
                     'validation/main/valid/recall20',
                     'validation/main/test/mrr5',
                     'validation/main/test/recall5',
                     'validation/main/test/mrr10',
                     'validation/main/test/recall10',
                     'validation/main/test/mrr15',
                     'validation/main/test/recall15',
                     'validation/main/test/mrr20',
                     'validation/main/test/recall20',
                     'elapsed_time']),
                    trigger=(1, 'iteration'))
                # trainer.extend(training.extensions.Evaluator(valid_iter, recsys, eval_func=lambda batch: evaluates(valid_dataset,test_dataset,test_batchsize, recsys), converter=converter, device=device), trigger=(1, 'epoch'))
                # trainer.extend(training.extensions.Evaluator(valid_iter, recsys, eval_func=lambda batch: evaluate(valid_dataset,test_batchsize, recsys,prefix='valid'), converter=converter, device=device), trigger=(10, 'iteration'))
                trainer.extend(training.extensions.Evaluator(test_iter, self.recsys,
                                                             eval_func=lambda batch: evaluate(self.test_dataset,
                                                                                              test_batchsize, self.recsys,
                                                                                              prefix='test'),
                                                             converter=converter, device=device), trigger=(1, 'epoch'))

                trainer.extend(
                    training.extensions.snapshot_object(self.recsys, model_name + '.model.{.updater.iteration}.npz'),
                    trigger=(1, 'epoch'))
                trainer.extend(
                    training.extensions.snapshot_object(optimizer, model_name + '.optimizer.{.updater.iteration}.npz'),
                    trigger=(1, 'epoch'))
                trainer.extend(lambda trainer: change_alpha(trainer), trigger=(3, 'epoch'))
                trainer.run()

    # def evaluate(self, test_dataset, batch_size, model, prefix='test'):
    #     pointer = 0
    #     eval_results = []
    #     while pointer < len(test_dataset):
    #         end = len(test_dataset) if pointer + batch_size >= len(test_dataset) else pointer + batch_size
    #         batch = test_dataset[pointer:end]
    #         results = model.test(batch)
    #         for i in range(len(batch)):
    #             eval_results.append([results[i], batch[i][1]])
    #         pointer += batch_size
    #
    #     mrr5 = mrr(eval_results, 5)
    #     recall5 = recall(eval_results, 5)
    #     reporter.report({prefix + '/mrr5': mrr5}, model)
    #     reporter.report({prefix + '/recall5': recall5}, model)
    #
    #     mrr10 = mrr(eval_results, 10)
    #     recall10 = recall(eval_results, 10)
    #     reporter.report({prefix + '/mrr10': mrr10}, model)
    #     reporter.report({prefix + '/recall10': recall10}, model)
    #
    #     mrr15 = mrr(eval_results, 15)
    #     recall15 = recall(eval_results, 15)
    #     reporter.report({prefix + '/mrr15': mrr15}, model)
    #     reporter.report({prefix + '/recall15': recall15}, model)
    #
    #     mrr20 = mrr(eval_results, 20)
    #     recall20 = recall(eval_results, 20)
    #     reporter.report({prefix + '/mrr20': mrr20}, model)
    #     reporter.report({prefix + '/recall20': recall20}, model)
    #     return mrr5, recall5, mrr10, recall10, mrr15, recall15, mrr20, recall20

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False,
                     type='view'):
        with chainer.using_config('cudnn_deterministic', True):
            with chainer.using_config('use_cudnn', 'auto'):
                #print(chainer.config.show())

                #device = 3

                #item2id, id2item = load_item(file='data/lastfm/lastfm_items.artist.txt')

                #test_dataset = SessionCorpus(file_path='data/lastfm/lastfm_test.repeat.artist.txt',item2id=item2id).load()

                #test_dataset = SessionCorpus(test, item2id=item2id).load()

                test_batchsize = 1

                # model=RepeatNet(len(item2id), embed_size=100, hidden_size=100,joint_train=False)
                #model = AttRec(len(self.item2id), embed_size=100, hidden_size=100)
                #recsys = RecSys(model, 20)
                #serializers.load_npz('model/att_rec_100_100_lastfm.model.26274.npz', recsys)

                # evaluate_mode(test_dataset,test_batchsize,model)

                # recsys = RecSys(model, 10)
                # serializers.load_npz('model/att_rec.model.140033.npz', recsys)

                #evaluate(self.test_dataset, test_batchsize, self.recsys, prefix='test')


                eval_results = []
                #while self.pointer < len(self.test_dataset):
                end = len(self.test_dataset) if self.pointer + test_batchsize >= len(self.test_dataset) else self.pointer + test_batchsize
                batch = self.test_dataset[self.pointer:end]
                results = self.recsys.test(batch)
                for i in range(len(batch)):
                    eval_results.append([results[i], batch[i][1]])
                self.pointer += test_batchsize
                #print(str(eval_results[0][0]))
                #print(str(self.item2id.keys()))
                g = pd.DataFrame(data=np.asanyarray(eval_results[0][0]),
                                 index=self.item2id.keys())[0]
                return g

