#coding=utf-8
from algorithms.STAMP.util.BatchData import batch_all
from algorithms.STAMP.util.Formater import add_pad
from algorithms.STAMP.util.Bitmap import bitmap_by_padid
import numpy as np
import copy
import random


class batcher(object):
    '''
    seq2seqatt batcher.
    '''

    def __init__(
        self,
        samples,
        class_num = None,
        random=True
    ):
        '''
        the init funciton.
        '''
        # TODO
        self.batch_size = 0
        self.class_num = class_num
        self.len_dic={}
        # unpack the samples
        for sample in samples:
            len_key = len(sample.click_items)
            if len_key in self.len_dic:
                self.len_dic[len_key].append(sample)
            else:
                self.len_dic[len_key] =[sample]

        self.key_list = list(self.len_dic.keys())
        if random is True:
            self.rand_idx = np.random.permutation(len(self.key_list))
        else:
            self.rand_idx = range(0, len(self.key_list))
        self.idx = 0

    def has_next(self):
        '''
        is hasing next epoch. 
        '''
        if self.idx >= len(self.rand_idx):
            return False
        else:
            return True

    def next_batch(self):
        '''
        get the netxt batch_data.
        '''
        self.ids = []
        self.in_idxes = []
        self.out_idxes = []



        samplelist = self.len_dic[self.key_list[self.rand_idx[self.idx]]]
        random.shuffle(samplelist)
        for sample in samplelist:
            self.ids.append(sample.id)
            self.in_idxes.append(sample.in_idxes)
            self.out_idxes.append(sample.out_idxes)

        rins, lab, rinlens, rmaxlens, rinlens_float32 = batch_all(
            [self.in_idxes,self.out_idxes]
        )
        self.idx += 1
        in_idxes = rins[0]
        out_idxes = rins[1]


        # context bitmap.
        sent_bitmap = []
        # row sentence lengths.
        sequence_lengs = rinlens[0]
        seq_lens = []
        for x in range(len(sequence_lengs)):
            nl = sequence_lengs[x][0]
            seq_lens.append(nl)

        ret_data = {
            'batch_ids': self.ids,
            'in_idxes': in_idxes,
            'out_idxes': out_idxes,
            'seq_lens': seq_lens
        }
        return ret_data
