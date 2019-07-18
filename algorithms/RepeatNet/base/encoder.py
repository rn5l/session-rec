# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from algorithms.RepeatNet.base.attention import *
from algorithms.RepeatNet.base.selective_gate import *
from chainer.initializers import GlorotNormal
from chainer.initializers import Uniform
from algorithms.RepeatNet.base.n_step_gru import *


class NStepGRUEncoder(chainer.Chain):
    def __init__(self, item_size, embed_size, hidden_size):
        super(NStepGRUEncoder, self).__init__(
            xe=L.EmbedID(item_size, embed_size,initialW=GlorotNormal(), ignore_label=-1),
            gru=NStepGRU(1, embed_size, hidden_size, 0.5),
        )
        self.hidden_size = hidden_size

    def __call__(self, input_list,x_enable):
        batch_size = len(input_list)
        exs = []
        for i in range(batch_size):
            exs.append(self.xe(self.xp.array(input_list[i],dtype=self.xp.int32)))

        state_next, batch_h_list = self.gru(None, exs)
        batch_last_h = F.vstack([h[-1,:self.hidden_size] for h in batch_h_list])
        batch_seq_h = F.pad_sequence(batch_h_list)
        return batch_last_h,batch_seq_h


class NStepSelBiGRUEncoder(chainer.Chain):
    def __init__(self, item_size, embed_size, hidden_size):
        super(NStepSelBiGRUEncoder, self).__init__(
            xe=L.EmbedID(item_size, embed_size,initialW=GlorotNormal(), ignore_label=-1),
            gru=NStepBiGRU(1, embed_size, hidden_size, 0.5),
            sel=SelectiveGate(2 * hidden_size)
        )
        self.hidden_size = hidden_size

    def __call__(self, input_list,x_enable):
        batch_size = len(input_list)
        exs = []
        for i in range(batch_size):
            exs.append(self.xe(self.xp.array(input_list[i],dtype=self.xp.int32)))

        state_next, batch_h_list = self.gru(None, exs)
        f_h = F.vstack([h[-1,:self.hidden_size] for h in batch_h_list])
        b_h = F.vstack([h[0, self.hidden_size:] for h in batch_h_list])

        batch_seq_h = F.pad_sequence(batch_h_list)

        batch_last_h = F.concat([f_h, b_h], axis=1)

        batch_seq_h = self.sel(batch_seq_h, batch_last_h, x_enable)

        return f_h,b_h,batch_last_h,batch_seq_h