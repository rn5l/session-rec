import tensorflow as tf
from algorithms.STAMP.util.Randomer import Randomer

class LinearLayer_3dim():
    '''
    the linear translate basic_layer.
    w_shape = [] : the shape of the w. 
    stddev: the random initialize base. 
    active: the active function. 
    ret = x * w
    '''

    # wline.shape = [edim, 1]
    # bline.shape = [1]
    def __init__(
        self,
        w_shape=None,
        stddev=None,
        params=None,
        active='tanh'
    ):
        '''
        the initialize function.
        w_shape is the shape of the w param. if params is None, need. 
        staddev is the stddev of the tf.random_normal. if params is None, need. 
        params = {'wline':wline}, is use to assign the params.
        active is the active function.  
        '''
        self.w_shape = w_shape
        self.stddev = stddev
        if params is None:
            self.wline = tf.Variable(
                Randomer.random_normal(self.w_shape),
                # tf.random_uniform(self.w_shape, -0.0015, 0.035),
                trainable=True
            )
        else:
            self.wline = params['wline']
        self.active = active

    # res.shape = [batch_size, time_steps, w_shp1]
    # inputs.shape = [batch_size, time_steps, edim]
    def forward(self, inputs):
        '''
        count
        '''

        w_shp0 = tf.shape(self.wline)[0]
        w_shp1 = tf.shape(self.wline)[1]
        batch_size = tf.shape(inputs)[0]
        # w_line_3dim.shape = [batch_size, edim, edim]
        w_line_3dim = tf.reshape(
            tf.tile(self.wline, [batch_size, 1]),
            [batch_size, w_shp0, w_shp1]
        )
        # linear translate
        res = tf.matmul(inputs, w_line_3dim)
        return res
