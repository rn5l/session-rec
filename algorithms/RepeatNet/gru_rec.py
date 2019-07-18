# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from algorithms.RepeatNet.base.encoder import *
from algorithms.RepeatNet.base.decoder import *
from algorithms.RepeatNet.base.utils import *

class GruRec(chainer.Chain):
    def __init__(self, item_size,embed_size, hidden_size):
        super(GruRec, self).__init__(
            enc=NStepGRUEncoder(item_size,embed_size, hidden_size),
            dec=GruDecoder(item_size),
        )

    def predict(self,input_list):
        x_enable = chainer.Variable(self.xp.array(mask(input_list)))
        batch_last_h, batch_seq_h = self.enc(input_list, x_enable)
        return self.dec(batch_last_h),

    def train(self,input_list,output_list):
        predicts=self.predict(input_list)[0]
        groundtruths = chainer.Variable(self.xp.array(output_list, dtype=self.xp.int32).reshape(-1,))
        # groundtruths=F.reshape(groundtruths,(-1,1))
        loss = F.softmax_cross_entropy(predicts, groundtruths, normalize=True, reduce='mean')
        return loss
