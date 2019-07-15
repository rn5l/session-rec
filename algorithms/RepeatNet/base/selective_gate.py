# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import GlorotNormal

class SelectiveGate(chainer.Chain):
    def __init__(self, hidden_size):
        super(SelectiveGate, self).__init__(
            xh=L.Linear(in_size=None,out_size=hidden_size,initialW=GlorotNormal(),nobias=True),
            hh=L.Linear(in_size=None,out_size=hidden_size,initialW=GlorotNormal(),nobias=True),
        )
        self.hidden_size = hidden_size

    def __call__(self,batch_seq_h, batch_h,enable):
        batch_size = batch_seq_h.shape[0]
        seq_size = batch_seq_h.shape[1]
        matp = F.expand_dims(self.xh(batch_h), axis=1)
        matp = F.broadcast_to(matp, (batch_size, seq_size, self.hidden_size))

        ab = F.reshape(batch_seq_h,(batch_size * seq_size, -1))
        wab = self.hh(ab)
        wab = F.reshape(wab, (batch_size, seq_size, -1))

        sGate=F.sigmoid(wab + matp)
        enable = F.expand_dims(enable, axis=2)
        enable = F.broadcast_to(enable, sGate.shape)
        sGate = F.where(enable, sGate, self.xp.zeros(sGate.shape, dtype=self.xp.float32))
        return batch_seq_h*sGate