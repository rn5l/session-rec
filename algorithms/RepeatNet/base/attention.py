# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import GlorotNormal

class Attention(chainer.Chain):
    def __init__(self, hidden_size):
        super(Attention, self).__init__(
            xh=L.Linear(in_size=None,out_size=hidden_size,initialW=GlorotNormal(),nobias=True),
            hh=L.Linear(in_size=None,out_size=hidden_size,initialW=GlorotNormal(),nobias=True),
            hw=L.Linear(hidden_size, 1,initialW=GlorotNormal(),nobias=True),
        )
        self.hidden_size=hidden_size

    def before_softmax_att_weights(self,batch_seq_h, batch_h,enable):
        batch_size = batch_seq_h.shape[0]
        seq_size = batch_seq_h.shape[1]
        # expand xh(h) to size of a matrix of encoder outputs
        matp = F.expand_dims(self.xh(batch_h), axis=1)
        matp = F.broadcast_to(matp, (batch_size, seq_size, self.hidden_size))
        # make a matrix of encoder (batch_size * sent_size, hidden_size)
        ab = F.reshape(batch_seq_h, (batch_size * seq_size, -1))
        # apply a weight vector to a encoder matrix
        wab = self.hh(ab)
        wab = F.reshape(wab, (batch_size, seq_size, -1))
        # apply a weight vector after calculating elementwise addtion (enc matrix + prev hidden output)
        e = self.hw(F.reshape(F.tanh(wab + matp), (batch_size * seq_size, -1)))
        e = F.reshape(e, (batch_size, seq_size))
        return e

    def att_weights(self,batch_seq_h, batch_h,enable):
        e=self.before_softmax_att_weights(batch_seq_h, batch_h,enable)
        e = F.where(enable, e, self.xp.ones(e.shape, dtype=self.xp.float32) * -float('inf'))
        # weights how amount enc vector affects
        att = F.softmax(e)
        return att


    def __call__(self,batch_seq_h, batch_h,enable):
        batch_size = batch_seq_h.shape[0]
        att = self.att_weights(batch_seq_h, batch_h,enable)
        c = F.batch_matmul(F.swapaxes(batch_seq_h, 1,2), att)
        return F.reshape(c, (batch_size, -1))
