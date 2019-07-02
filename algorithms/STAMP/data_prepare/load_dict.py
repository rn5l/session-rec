import numpy as np
import copy

# def load_glove(pretrain_file, word2idx, pad_idx = 0, edim = 300, init_std = 0.05):
#     emb_dict = np.random.normal(0, init_std, [len(word2idx), edim])
#     emb_dict[pad_idx] = [0.0] * edim
#     incnt = 0
#     inids = []
#     inids.append(0)
#     not_inwords = []
#     with open(pretrain_file, 'r') as f:
#         for line in f:
#             content = line.strip().split()
#             if content[0] in word2idx:
#                 emb_dict[word2idx[content[0]]] = np.array(map(float, content[1:]))
#                 incnt += 1
#                 inids.append(word2idx[content[0]])
#     print "not in the pretrain embedding glove word num:" + str(len(word2idx) - incnt)
#     for w in word2idx:
#         if word2idx[w] not in inids:
#             not_inwords.append(copy.deepcopy(w))
#     print not_inwords
#
#     # for w in word2idx:
#     #     if word2idx[w] not in inids:
#     #         print w
#     return emb_dict
#
#
# def load_ssweu(pretrain_file, word2idx, pad_idx=0, edim=50, init_std=0.05):
#     emb_dict = np.random.normal(0, init_std, [len(word2idx), edim])
#     emb_dict[pad_idx] = [0.0] * edim
#     incnt = 0
#     inids = []
#     inids.append(0)
#     not_inwords = []
#     with open(pretrain_file, 'r') as f:
#         for line in f:
#             content = line.strip().split()
#             if content[0] in word2idx:
#                 emb_dict[word2idx[content[0]]] = np.array(map(float, content[1:]))
#                 incnt += 1
#                 inids.append(word2idx[content[0]])
#     print "not in the pretrain embedding sswe word num:" + str(len(word2idx) - incnt)
#     for w in word2idx:
#         if word2idx[w] not in inids:
#             not_inwords.append(copy.deepcopy(w))
#     print not_inwords
#
#     # for w in word2idx:
#     #     if word2idx[w] not in inids:
#     #         print w
#     return emb_dict


def load_random( word2idx, pad_idx=0, edim=300, init_std=0.05):
    # sigma = np.sqrt(2./(len(word2idx)-1))
    emb_dict = np.random.normal(0, init_std, [len(word2idx), edim])
    # emb_dict = np.random.randn(*(len(word2idx),edim))*sigma
    emb_dict[pad_idx] = [0.0] * edim
    # diag = np.ones(n_items)
    # emb_dict = np.diag(diag)
    return emb_dict

