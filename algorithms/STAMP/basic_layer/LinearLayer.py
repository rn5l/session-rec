#coding=utf-8
import tensorflow as tf
from algorithms.STAMP.util.Randomer import Randomer

class LinearLayer(object):

    def __init__(self, w_shape, stddev = None, params=None):
        '''
        :param w_shape: [input_dim, output_dim]
        :param stddev: 用于初始化
        :param params: 从外界制定参数
        '''
        if params is None:
            self.w = tf.Variable(
                Randomer.random_normal(w_shape),
                trainable=True
            )
        else:
            self.w = params['w']
    def forward(self, inputs):
        '''
        count
        '''
        # batch_size = tf.shape(inputs)[0]
        # w_shp0 = tf.shape(self.w)[0]
        # w_shp1 = tf.shape(self.w)[1]
        # w_line_3dim.shape = [batch_size, edim, edim]
        # w_line_3dim = tf.reshape(
        #     tf.tile(self.w, [batch_size, 1]),
        #     [batch_size, w_shp0, w_shp1]
        # )
        # linear translate
        res = tf.matmul(inputs, self.w)
        return res