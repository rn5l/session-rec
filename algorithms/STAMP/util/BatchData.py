import copy

def batch_range(
    batch_size=0,
    nidx=0,
    lsamps=0,
    rand_idx=[],
    class_num=0,
    labels=[],
    ids = [],
    inputs=[[]],
):
    '''
    get the batch data. 
    batch_size: the max size of this batch. 
    nidx: now the index of the data which has not been taken. 
    lsamps: the total len of the samples. 
    lsamps: int
    nidx: int
    class_num: int

    inputs = [contexts, aspects, ...]
    contexts: the context data. 
    contexts.shape = [-1, -1, edim], 
    contexts = [[sentence[word ebedding],[],[],...],[],...]
    the first -1 means the all samples. 
    the second -1 means the different size of the sentence. 

    aspects: the aspect data. 
    labels: the label data.
    labels: shape = [len(samples)] 
    rand_idx: the random indexes of the data.  [2, 1, 4, 5, 3 ...]
    class_num: the total number of the classes. 

    ret: 
    ctx: the context data has been taken out. 
    asp: the aspect data has been taken out. 
    lab: the label data has been taken out. 
    asp_lens: the aspects' lens, all of the been taken out aspects. 
    asp_len: the max len of the aspect, use for the format function to padding. 
    mem_size: the max len of the context. 
    nidx: now has not taken out data's index. 
    '''
    rins = []  # the r different inputs of this batch.
    for _ in range(len(inputs)):
        rins.append([])
    # shape= [batchs_size, class_num]. type: float32. the label of this batch.
    lab = []
    ret_ids = []
    rinlens = []  # the r different inputs' length of each sample
    rinlens_float32 = []
    for _ in range(len(inputs)):
        rinlens.append([])
        rinlens_float32.append([])
    rmaxlen = []  # the r different inputs' max len of the sample.
    for _ in range(len(inputs)):
        rmaxlen.append(0)

    for bs in range(batch_size):  # get this batch data.
        if nidx >= lsamps:
            break
        for i in range(len(inputs)):
            rins[i].append(copy.deepcopy(inputs[i][rand_idx[nidx]]))
            crt_len = len(rins[i][-1])
            rinlens[i].append([crt_len])
            rinlens_float32[i].append([crt_len])
            if rmaxlen[i] < crt_len:
                rmaxlen[i] = crt_len

        # the lab should be float32.
        crt_lab = [0.0] * class_num

        crt_lab[labels[rand_idx[nidx]]] = 1.0
        lab.append(crt_lab)
        ret_ids.append(ids[rand_idx[nidx]])
        nidx += 1
    # cast the asp_lens to float32. use for mean the aspect.
    # beacuse the aspects has been pad.
    for j in range(len(inputs)):
        rinlens_float32[j] = [[float(i[0])] for i in rinlens_float32[j]]

    return rins, lab, ret_ids, rinlens, rmaxlen, nidx, rinlens_float32


def batch_all(inputs=[[]], labels=[], class_num=0):
    '''
    read all the data into the ctx, asp, lab. 
    '''
    rins = []  # the r different inputs of this batch.
    for _ in range(len(inputs)):
        rins.append([])
    # shape= [batchs_size, class_num]. type: float32. the label of this batch.
    lab = []
    rinlens = []  # the r different inputs' length of each sample
    rinlens_float32 = []
    for _ in range(len(inputs)):
        rinlens.append([])
        rinlens_float32.append([])
    rmaxlen = []  # the r different inputs' max len of the sample.
    for _ in range(len(inputs)):
        rmaxlen.append(0)

    for j in range(len(inputs[0])):  # get this batch data.
        for i in range(len(inputs)):
            rins[i].append(copy.deepcopy(inputs[i][j]))
            crt_len = len(rins[i][-1])
            rinlens[i].append([crt_len])
            rinlens_float32[i].append([crt_len])
            if rmaxlen[i] < crt_len:
                rmaxlen[i] = crt_len
        if class_num == 0:
            lab = None
        else:
            # the lab should be float32.
            crt_lab = [0.0] * class_num
            crt_lab[labels[j]] = 1.0
            lab.append(crt_lab)
    # cast the asp_lens to float32. use for mean the aspect.
    # beacuse the aspects has been pad.
    for j in range(len(inputs)):
        rinlens_float32[j] = [[float(i[0])] for i in rinlens_float32[j]]

    return rins, lab, rinlens, rmaxlen, rinlens_float32
