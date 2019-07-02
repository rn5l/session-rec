import numpy as np

def bitmap_by_padid(inputs, padid):
    '''
    inputs: the tensor consists of ids. 
    padid: the pad id. 
    generate the bitmap according to the inputs and padid. 
    the shape of bitmap is same as inputs. 
    '''
    ret = []
    if len(np.shape(inputs)) == 1:
        for idx in inputs:
            if idx == padid:
                ret.append(float(0))
            else:
                ret.append(float(1))
    else:
        for ip in inputs:
            ret.append(bitmap_by_padid(ip, padid))
    return ret
