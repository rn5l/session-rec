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
from algorithms.RepeatNet.base.utils import *
from chainer.initializers import GlorotNormal
from chainer.initializers import Uniform
from algorithms.RepeatNet.base.function import *


class GruDecoder(chainer.Chain):
    def __init__(self, item_size):
        super(GruDecoder, self).__init__(
            fy=L.Linear(None, item_size,initialW=GlorotNormal(),nobias=True)
        )

    def __call__(self, x_last_h):
        h=F.dropout(x_last_h,.5)
        h = self.fy(h)
        return h

class AttDecoder(chainer.Chain):
    def __init__(self, item_size, hidden_size):
        super(AttDecoder, self).__init__(
            att=Attention(hidden_size),
            fy=L.Linear(None, item_size,initialW=GlorotNormal(),nobias=True)
        )

    def __call__(self, x_last_h,batch_x_seq_h,x_enable):
        x_last_h = F.dropout(x_last_h, .5)
        batch_x_seq_h = F.dropout(batch_x_seq_h, .5)

        enc = self.att(batch_x_seq_h, x_last_h, x_enable)
        h=F.concat([x_last_h,enc],axis=1)
        h = self.fy(h)
        return h

class NoAttReDecoder(chainer.Chain):
    def __init__(self, item_size, hidden_size):
        super(NoAttReDecoder, self).__init__(
            re=L.Linear(None,2,initialW=GlorotNormal(),nobias=True),
            re_att=Attention(2*hidden_size),
            fy=L.Linear(None, item_size,initialW=GlorotNormal(),nobias=True)
        )

    def __call__(self, x_last_h,input_list,batch_x_seq_h,x_enable):
        x_last_h=F.dropout(x_last_h,.5)
        batch_x_seq_h=F.dropout(batch_x_seq_h,.5)

        p_re=F.softmax(self.re(x_last_h),axis=1)
        p=F.swapaxes(p_re,0,1)
        p_e=F.get_item(p,0)
        p_r = F.get_item(p, 1)

        p_i_e=self.fy(F.dropout(x_last_h, .5))
        p_i_e=expore(p_i_e, input_list)
        p_i_e = F.softmax(p_i_e,axis=1)

        p_i_r = self.re_att.att_weights(batch_x_seq_h, x_last_h, x_enable)
        p_i_r=repeat(p_i_r, input_list, x_enable, p_i_e.shape)

        return p_i_r*F.broadcast_to(F.expand_dims(p_r,1),p_i_r.shape), p_i_e*F.broadcast_to(F.expand_dims(p_e,1),p_i_e.shape),p_re


class AttReDecoder(chainer.Chain):
    def __init__(self, item_size, hidden_size):
        super(AttReDecoder, self).__init__(
            re=L.Linear(None,2,initialW=GlorotNormal(),nobias=True),
            re_att=Attention(hidden_size),
            r_att=Attention(hidden_size),
            e_att=Attention(hidden_size),
            fy=L.Linear(None, item_size,initialW=GlorotNormal(),nobias=True)
        )

    def __call__(self, x_last_h,input_list,batch_x_seq_h,x_enable):
        x_last_h=F.dropout(x_last_h,.5)
        batch_x_seq_h=F.dropout(batch_x_seq_h,.5)

        h = self.re_att(batch_x_seq_h, x_last_h, x_enable)
        p_re=F.softmax(self.re(h),axis=1)
        p=F.swapaxes(p_re,0,1)
        p_e=F.get_item(p,0)
        p_r = F.get_item(p, 1)

        h = self.e_att(batch_x_seq_h, x_last_h, x_enable)
        p_i_e=self.fy(F.concat([h,x_last_h],axis=1))
        p_i_e=expore(p_i_e, input_list)
        p_i_e = F.softmax(p_i_e,axis=1)

        p_i_r = self.r_att.att_weights(batch_x_seq_h, x_last_h, x_enable)
        p_i_r=repeat(p_i_r, input_list, x_enable, p_i_e.shape)

        return p_i_r*F.broadcast_to(F.expand_dims(p_r,1),p_i_r.shape), p_i_e*F.broadcast_to(F.expand_dims(p_e,1),p_i_e.shape),p_re








