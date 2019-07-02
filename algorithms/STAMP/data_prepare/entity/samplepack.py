#coding=utf-8

class Samplepack(object):
    def __init__(self):
        self.samples = []
        self.id2sample = {}

    def init_id2sample(self):
        if self.samples is None:
            raise Exception("Samples is None.", self.samples)
        for sample in self.samples:
            self.id2sample[sample.id] = sample

    def pack_preds(self, preds, ids):
        '''
        preds和ids是list，二者顺序一一对应
        :param preds:
        :param ids:
        :return:
        '''
        # print preds
        # print ids
        for i in range(len(ids)):
            self.id2sample[ids[i]].pred.append(preds[i])
    def flush(self):
        for sample in self.samples:
            sample.pred = []

    #
    def update_best(self):
        for sample in self.samples:
            sample.best_pred = sample.pred
    #
    # def pack_memories(self, memories, ids, has_eos):
    #     for i in range(len(ids)):
    #         sample = self.id2sample[ids[i]]
    #         memory = memories[i]
    #         if has_eos:
    #             memory = memory[:-1]
    #         sample.memory = memory

    def pack_ext_matrix(self, name, matrixes, ids):
        # matrixes维度为3,第0维对应于样本s. len(matrixes)表示样本个数.
        for i in range(len(ids)):
            # print(matrixes[i])
            self.id2sample[ids[i]].ext_matrix[name].append(matrixes[i])

    def transform_ext_matrix(self, matrixes):
        #输入为[hops,batch,time_tep],输出为[batch,hop,timestep]
        tra_matrix = []
        for x in range(len(matrixes[0])):
            tra_matrix.append([])
        for i in range(len(tra_matrix)):
            for x in matrixes:
                tra_matrix[i].append(x[i])
        return tra_matrix