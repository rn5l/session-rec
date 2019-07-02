import tensorflow as tf


def pooler(inputs, pool_type, axis=1, **kwargs):
    '''
    the pool function. 
    '''
    if pool_type == 'mean':
        return mean_pool(inputs, kwargs['sequence_length'], axis)
    elif pool_type == 'max':
        return max_pool(inputs, axis)
    elif pool_type == 'sum':
        return sum_pool(inputs, axis)


def mean_pool(inputs, sequence_length=None, axis=1):
    '''
    the mean pool function. 
    inputs.shape = [batch_size, timestep_size, edim]
    sequence_length = [batch_size, 1]
    sequence_length = [[len(sequence)], [], ...]
    '''
    if sequence_length is None:
        return tf.reduce_mean(inputs, axis)
    else:
        return tf.div(tf.reduce_sum(inputs, axis), sequence_length)


def max_pool(inputs, axis=1):
    '''
    the max pool function. 
    '''
    return tf.reduce_max(inputs, axis)


def min_pool(inputs, axis=1):
    '''
    the min pool function. 
    '''
    return tf.reduce_min(inputs, axis)


def sum_pool(inputs, axis=1):
    '''
    the add pool function. 
    '''
    return tf.reduce_sum(inputs, axis)
