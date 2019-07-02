# coding=utf-8

import tensorflow as tf


def last_relevant(inputs, lengths):
    '''从rnn的输出中取出最后一个词的输出

    Args:
        :type inputs: tensor, shape = [batch_size, time_step, edim]
        :param inputs: rnn的输出，

        :type lengths: tensor, shape = [batch_size]
        :param lengths: rnn中各个输入的真实长度

    Returns:
        一个shape=[batch_size, edim]的张量，第0维的每个元素是对应的Rnn的最后的输出。
    '''
    lengths = tf.cast(lengths, dtype=tf.int32)
    batch_size = tf.shape(inputs)[0]
    time_step = tf.shape(inputs)[1]
    edim = tf.shape(inputs)[2]
    index = tf.range(0, batch_size) * time_step + (lengths - 1)
    flat = tf.reshape(inputs, [-1, edim])
    relevant = tf.gather(flat, index)
    return relevant


def last_relevant1(inputs, lengths):
    '''从rnn的输出中取出最后一个词的输出

    Args:
        :type inputs: tensor, shape = [batch_size, time_step, edim]
        :param inputs: rnn的输出，

        :type lengths: tensor, shape = [batch_size]
        :param lengths: rnn中各个输入的真实长度

    Returns:
        一个shape=[batch_size, edim]的张量，第0维的每个元素是对应的Rnn的最后的输出。
    '''
    ips = tf.unstack(inputs, axis=0)
    lens = tf.unstack(lengths, axis=0)
    reles = []
    for i in range(len(lens)):
        reles.append(tf.gather(ips[i], tf.range(lens[i] - 1, lens[i])))
    reles = tf.concat(reles, axis=1)

    return reles


def relevants(inputs, lengths, is_concat=True):
    '''从rnn的输出中取出最后一个词的输出

    Args:
        :type inputs: tensor, shape = [batch_size, time_step, edim]
        :param inputs: rnn的输出，

        :type lengths: tensor, shape = [batch_size]
        :param lengths: rnn中各个输入的真实长度

    Returns:
        一个shape=[batch_size * real_len, edim]的张量，第0维的每个元素是对应的Rnn的最后的输出。
    '''
    ips = tf.unstack(inputs, axis=0)
    lens = tf.unstack(lengths, axis=0)
    reles = []
    for i in range(len(lens)):
        reles.append(tf.gather(ips[i], tf.range(0, lens[i])))
    if is_concat:
        reles = tf.concat(reles, axis=0)
    return reles

