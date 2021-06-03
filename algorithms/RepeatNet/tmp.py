from algorithms.RepeatNet.base.corpus import *
import numpy as np
import chainer.links as L
import chainer.functions as F


# probability = np.array([0,1,2,1],dtype=np.int32)
# vv = np.array([[0,1,2,1],[0,1,2,1],[0,1,2,1],[0,1,2,1]],dtype=np.float32)
#
# slices=[]
# for i, v in enumerate(probability):
#     slices.append((i,v))
# print slices[:2]
# print F.get_item(vv,slices[:2])

# indices = np.argsort([-p for p in probability]).astype(dtype=np.int32)
# results=[result[:2] for result in indices]
# print results


# n=0
# o=0
# with codecs.open('data/diginetica/digi_test.txt', encoding='utf-8') as f:
#     for line in f:
#         lines = line.strip('\n').strip('\r').split('\t')
#         input = ast.literal_eval(lines[0])
#         if int(lines[1]) not in input:
#             n+=1
#         else:
#             o+=1
#
# print n/float(n+o),o/float(n+o),n,o

ff = codecs.open('data/yoo_1_4/test_1_over_4.repeat.txt', encoding='utf-8', mode='w')
fff = codecs.open('data/yoo_1_4/test_1_over_4.nonrepeat.txt', encoding='utf-8', mode='w')
with codecs.open('data/yoo_1_4/test_1_over_4.txt', encoding='utf-8') as f:
    for line in f:
        lines = line.strip('\n').strip('\r').split('\t')
        input = ast.literal_eval(lines[0])
        output=int(lines[1])
        if output in input:
            ff.write(line)
        else:
            fff.write(line)

ff.close()
fff.close()



# item2id,id2item=load_item(file='data/yoo_items_1_4.txt')
# train_set=SessionCorpus(file_path='data/yoo_test_1_4.txt',item2id=item2id).load()

# items=set()
# with codecs.open('data/diginetica/digi_train.txt', encoding='utf-8') as f:
#     for line in f:
#         lines = line.strip('\n').strip('\r').split('\t')
#         input = ast.literal_eval(lines[0])
#         for item in input:
#             if item not in items:
#                 items.add(item)
#         if lines[1] not in items:
#             items.add(int(lines[1]))
#
# with codecs.open('data/diginetica/digi_valid.txt', encoding='utf-8') as f:
#     for line in f:
#         lines = line.strip('\n').strip('\r').split('\t')
#         input = ast.literal_eval(lines[0])
#         for item in input:
#             if item not in items:
#                 items.add(item)
#         if lines[1] not in items:
#             items.add(int(lines[1]))
#
# with codecs.open('data/diginetica/digi_test.txt', encoding='utf-8') as f:
#     for line in f:
#         lines = line.strip('\n').strip('\r').split('\t')
#         input = ast.literal_eval(lines[0])
#         for item in input:
#             if item not in items:
#                 items.add(item)
#         if lines[1] not in items:
#             items.add(int(lines[1]))
#
# f = codecs.open('data/diginetica/digi_items.txt', encoding='utf-8', mode='w')
# for k in items:
#     f.write(str(k)+ os.linesep)
# f.close()


