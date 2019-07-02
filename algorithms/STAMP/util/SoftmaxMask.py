import tensorflow as tf

def normalizer(norm_type, inputs, seq_mask, axis = 1):
    switch = {
        'softmax': softmax_mask,
        'alpha': alpha_mask,
        'sigmoid': sigmoid_mask,
        'sigmoid2' : sigmoid2_mask,
        'none': none_mask,
    }
    func = switch.get(norm_type, softmax_mask)
    return func(inputs, seq_mask, axis)
def none_mask(inputs, seq_mask, axis=1):
    return inputs*seq_mask

def softmax_mask(inputs, seq_mask, axis=1):
    '''
    The softmax mask, use to remove the pad. 

    :type inputs: tensor
    :param inputs: the inputs should use to count the softmax. 

    :type seq_mask: tensor
    :param seq_mask: the mask tensor consists of 0 or 1, should
    has the same shape with inputs. 

    :type axis: int 
    :param axis: the axis of the softmax on. 
    '''
    inputs = tf.cast(inputs, tf.float32)
    inputs = inputs * seq_mask
    # max_nums = tf.reduce_max(inputs, axis, keep_dims=True)
    inputs = tf.exp(inputs)
    inputs = inputs * seq_mask
    _sum = tf.reduce_sum(inputs, axis=axis, keep_dims=True) + 1e-9
    return inputs / _sum

def alpha_mask(inputs, seq_mask, axis = 1):
    inputs = tf.cast(inputs, tf.float32)
    inputs = inputs * seq_mask
    max_nums = tf.reduce_max(inputs, axis, keep_dims=True)
    inputs = tf.exp(inputs - max_nums)
    outputs = inputs * seq_mask
    return outputs

def sigmod_mask(inputs, seq_mask, axis = 1):
    return sigmoid_mask(inputs, seq_mask, axis)

def sigmoid_mask(inputs, seq_mask, axis = 1):
    inputs = tf.cast(inputs, tf.float32)
    inputs = inputs * seq_mask
    inputs = tf.sigmoid(inputs)
    outputs = inputs * seq_mask
    return outputs

def sigmoid2_mask(inputs, seq_mask, axis = 1):
    outputs = sigmoid_mask(inputs, seq_mask, axis)
    outputs /= (tf.reduce_sum(outputs, axis, keep_dims = True) / 2)
    return outputs