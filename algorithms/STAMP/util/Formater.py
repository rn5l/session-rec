def add_pad(inputs = [], max_lens = [], pad_idx = 0):
    '''
    Format the input to tensor. 

    inputs.shape = [n]
    max_lens.shape = [n]

    inputs = [nip1, nip2, ..., nipn],
    nipi.shape = [batch_size, len(sentence)],
    nipi = [[id0, id1, id2, ...], [id0, id1, id2, ...], ...]

    max_lens = [nml1, nml2, ..., nmln]
    max_lens.shape = [n]
    nml1 = int. means the max length of the nipi's sentences.

    the pad is use on the second dim of the nipi. 

    pad_idx: the padding word's id. 
    '''
    if len(inputs) != len(max_lens):
        print("the max_lens.len not equal the inputs.len")
        return
    for i in range(len(inputs)):
        nips = inputs[i]
        nml = max_lens[i]
        for nip in nips:
            crt_len = len(nip)
            for _ in range(nml - crt_len):
                nip.append(pad_idx)
