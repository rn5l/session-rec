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
#print('Cupy Version: ', str(cuda.cupy.__version__))
#print('CuDNN Version: ',str(chainer._cudnn_version))


def evaluates_mode(valid_dataset,test_dataset,batch_size,model):
    evaluate_mode(valid_dataset, batch_size, model)
    evaluate_mode(test_dataset, batch_size, model)

def evaluate_mode(test_dataset, batch_size, model):
    pointer = 0
    eval_results = []
    while pointer < len(test_dataset):
        end = len(test_dataset) if pointer + batch_size >= len(test_dataset) else pointer + batch_size
        batch = test_dataset[pointer:end]
        input_list=[b[0] for b in batch]
        results = model.predict(input_list)
        for i in range(len(batch)):
            eval_results.append([results[1][i], batch[i][1][0] in batch[i][0]])
        pointer += batch_size

    return accuracy(eval_results)

def accuracy(eval_results):
    correct=0
    for one in eval_results:
        if one[1] and one[0][1].data>one[0][0].data:
            correct+=1
    print(float(correct)/len(eval_results))
    return float(correct)/len(eval_results)

if __name__ == '__main__':
    with chainer.using_config('cudnn_deterministic', True):
        with chainer.using_config('use_cudnn', 'auto'):
            print(chainer.config.show())

            device = 3

            item2id, id2item = load_item(file='data/lastfm/lastfm_items.artist.txt')

            test_dataset = SessionCorpus(file_path='data/lastfm/lastfm_test.repeat.artist.txt', item2id=item2id).load()

            test_batchsize = 1024

            # model=RepeatNet(len(item2id), embed_size=100, hidden_size=100,joint_train=False)
            model = AttRec(len(item2id), embed_size=100, hidden_size=100)
            recsys = RecSys(model, 20)
            serializers.load_npz('model/att_rec_100_100_lastfm.model.26274.npz', recsys)


            # evaluate_mode(test_dataset,test_batchsize,model)

            # recsys = RecSys(model, 10)
            # serializers.load_npz('model/att_rec.model.140033.npz', recsys)

            print(evaluate(test_dataset, test_batchsize, recsys, prefix='test'))

            test_dataset = SessionCorpus(file_path='data/lastfm/lastfm_test.nonrepeat.artist.txt', item2id=item2id).load()

            print(evaluate(test_dataset, test_batchsize, recsys, prefix='test'))
