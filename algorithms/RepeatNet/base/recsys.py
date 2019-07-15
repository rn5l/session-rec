import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class RecSys(chainer.Chain):
    def __init__(self, model,top_k):
        self.top_k=top_k
        chainer.Chain.__init__(self,
                               model=model
                               )

    def train(self,batch):
        input = []
        output=[]
        for i in range(len(batch)):
            input.append(batch[i][0])
            output.append(batch[i][1])

        with chainer.using_config('train', True):
            loss=self.model.train(input,output)
            sum_loss = 0
            if isinstance(loss, tuple):
                for i in range(len(loss)):
                    chainer.report({'loss' + str(i): loss[i].data}, self)
                    sum_loss += loss[i]
            else:
                sum_loss = loss

            chainer.report({'loss': sum_loss.data}, self)
            return sum_loss

    def test(self,batch):
        with cuda.Device(self._device_id):
            with chainer.using_config('train', False):
                input=[]
                for i in range(len(batch)):
                    input.append(batch[i][0])
                #print('input '+str(input))
                probability= self.model.predict(input)[0].data
                if not self._cpu:
                    probability = cuda.to_cpu(probability)
                #indices = np.argsort([-p for p in probability]).astype(dtype=np.int32)
                #results=[result[:self.top_k] for result in indices]
                #return results
                return probability

    def __call__(self,batch):
        with cuda.Device(self._device_id):
            with chainer.using_config('train', True):
                return self.train(batch)
