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
from algorithms.RepeatNet.base.function import *


class NoAttRepeatNet(chainer.Chain):
    def __init__(self, item_size,embed_size, hidden_size,joint_train=False):
        self.joint_train=joint_train
        super(NoAttRepeatNet, self).__init__(
            enc=NStepGRUEncoder(item_size,embed_size, hidden_size),
            dec=NoAttReDecoder(item_size, hidden_size),
        )

    def predict(self,input_list):
        x_enable = chainer.Variable(self.xp.array(mask(input_list)))
        batch_last_h, batch_seq_h = self.enc(input_list, x_enable)
        p_r, p_e, p = self.dec(batch_last_h, input_list, batch_seq_h, x_enable)

        return p_r + p_e, p

    def train(self,input_list,output_list):
        predicts, p = self.predict(input_list)

        slices = self.xp.zeros(predicts.shape, dtype=self.xp.int32) > 0
        if self.joint_train:
            p_slices = self.xp.zeros(p.shape, dtype=self.xp.int32) > 0
        for i, v in enumerate(output_list):
            slices[i, v] = True
            if self.joint_train:
                if v in input_list[i]:
                    p_slices[i, 1] = True
                else:
                    p_slices[i, 0] = True

        loss = -F.sum(F.log(F.get_item(predicts, slices))) / len(input_list)
        if self.joint_train:
            p_loss = -F.sum(F.log(F.get_item(p, p_slices))) / len(input_list)
        if self.joint_train:
            return loss, p_loss
        else:
            return loss

class RepeatNet(chainer.Chain):
    def __init__(self, item_size,embed_size, hidden_size,joint_train=False):
        self.joint_train = joint_train
        super(RepeatNet, self).__init__(
            enc=NStepGRUEncoder(item_size,embed_size, hidden_size),
            dec=AttReDecoder(item_size, hidden_size),
        )

    def predict(self,input_list):
        x_enable = chainer.Variable(self.xp.array(mask(input_list)))
        batch_last_h, batch_seq_h = self.enc(input_list, x_enable)
        p_r,p_e,p= self.dec(batch_last_h, input_list,batch_seq_h, x_enable)

        return p_r+p_e,p

    def train(self,input_list,output_list):
        predicts,p=self.predict(input_list)

        slices=self.xp.zeros(predicts.shape, dtype=self.xp.int32)>0
        if self.joint_train:
            p_slices = self.xp.zeros(p.shape, dtype=self.xp.int32) > 0
        for i, v in enumerate(output_list):
            slices[i,v]=True
            if self.joint_train:
                if v in input_list[i]:
                    p_slices[i,1]=True
                else:
                    p_slices[i, 0] = True

        loss=-F.sum(F.log(F.get_item(predicts,slices)))/len(input_list)
        if self.joint_train:
            p_loss=-F.sum(F.log(F.get_item(p,p_slices)))/len(input_list)
        if self.joint_train:
            return loss,p_loss
        else:
            return loss
