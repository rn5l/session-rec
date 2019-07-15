import os
import importlib
os.environ["CHAINER_TYPE_CHECK"] = "0"
import sys
importlib.reload(sys)
sys.setdefaultencoding('utf-8')
from algorithms.RepeatNet.base.corpus import *
import pickle
import argparse
import random
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers,training,reporter,iterators
from chainer.training.updaters import MultiprocessParallelUpdater
from chainer.iterators import MultiprocessIterator
import codecs
import collections
from algorithms.RepeatNet.gru_rec import *
from algorithms.RepeatNet.att_rec import *
from algorithms.RepeatNet.repeat_net import *
from algorithms.RepeatNet.base.recsys import *

print('Chainer Version: '+chainer.__version__)
print('Cupy Version: ', cuda.cupy.__version__)
print('CuDNN Version: ',chainer._cudnn_version)

if __name__ == '__main__':
    with chainer.using_config('cudnn_deterministic', True):
        with chainer.using_config('use_cudnn', 'auto'):
            print(chainer.config.show())

            device = 0
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            cuda.cupy.random.seed(seed)

            item2id, id2item = load_item(file='data/diginetica/digi_items.txt')

            model_name = 'repeat_rec_100_100_digi'
            print(model_name, 'GPU: ',device)
            train_batchsize = 1024
            train_dataset = SessionCorpus(file_path='data/diginetica/digi_train.txt', item2id=item2id).load()
            # valid_dataset = SessionCorpus(file_path='data/yoo_1_4/valid_1_over_4.txt', item2id=item2id).load()
            test_dataset = SessionCorpus(file_path='data/diginetica/digi_valid.txt', item2id=item2id).load()
            test_batchsize = 1024

            # model=AttRec(len(item2id), embed_size=100, hidden_size=100)
            # model = NoAttRepeatNet(len(item2id), embed_size=100, hidden_size=100, joint_train=False)
            model=RepeatNet(len(item2id), embed_size=100, hidden_size=100,joint_train=False)
            recsys = RecSys(model, 20)
            # serializers.load_npz('model/repeat_rec_100_100_yoo.model.52012.npz', recsys)

            optimizer = optimizers.Adam(alpha=0.001)
            # optimizer = optimizers.AdaDelta(rho=0.95, eps=1e-6)
            optimizer.setup(recsys)
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
            test_iter = iterators.SerialIterator(test_dataset, batch_size=len(test_dataset), shuffle=False, repeat=False)

            def converter(batch, device):
                return batch
            updater = training.StandardUpdater(train_iter, optimizer, converter=converter, device=device)
            # updater = MultiprocessParallelUpdater(train_iters, optimizer, converter=converter, devices=devices)

            trainer = training.Trainer(updater)
            trainer.out = 'model/'
            trainer.extend(training.extensions.LogReport(trigger=(100, 'iteration')))
            trainer.extend(training.extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss','main/loss0','main/loss1','main/loss2',
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
                trigger=(100, 'iteration'))
            # trainer.extend(training.extensions.Evaluator(valid_iter, recsys, eval_func=lambda batch: evaluates(valid_dataset,test_dataset,test_batchsize, recsys), converter=converter, device=device), trigger=(1, 'epoch'))
            # trainer.extend(training.extensions.Evaluator(valid_iter, recsys, eval_func=lambda batch: evaluate(valid_dataset,test_batchsize, recsys,prefix='valid'), converter=converter, device=device), trigger=(10, 'iteration'))
            trainer.extend(training.extensions.Evaluator(test_iter, recsys, eval_func=lambda batch: evaluate(test_dataset,test_batchsize, recsys,prefix='test'), converter=converter, device=device), trigger=(1, 'epoch'))
            trainer.extend(training.extensions.snapshot_object(recsys, model_name+'.model.{.updater.iteration}.npz'), trigger=(1, 'epoch'))
            trainer.extend(training.extensions.snapshot_object(optimizer, model_name+'.optimizer.{.updater.iteration}.npz'), trigger=(1, 'epoch'))
            trainer.extend(lambda trainer: change_alpha(trainer), trigger=(3, 'epoch'))
            trainer.run()
