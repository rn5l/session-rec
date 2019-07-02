import tensorflow as tf
from algorithms.STAMP.basic_layer.LinearLayer_3dim import LinearLayer_3dim
from algorithms.STAMP.util.Activer import activer
from algorithms.STAMP.util.SoftmaxMask import normalizer


class FwNnAttLayer(object):
    '''
    The simple forward neural network attention model. 
    '''

    def __init__(self, edim, active='tanh', stddev=None, params=None, norm_type = 'softmax'):
        '''
        :type edim: int
        :param edim: the edim of the input embedding. 

        :type stddev: float
        :param stddev: the stddev use in the normal random. 

        :type params: dict
        :param params: the initialial params, {'wline_ctx':params, 'wline_asp':params}
        '''
        self.edim = edim
        self.active = active
        self.norm_type = norm_type
        if params == None:
            wline_asp = None
            wline_ctx = None
            wline_out = None
            wline_att_ca = None
        else:
            wline_asp = params['wline_asp']
            wline_ctx = params['wline_ctx']
            wline_out = params['wline_out']
            wline_att_ca = params['wline_att_ca']
        self.line_layer_asp = LinearLayer_3dim(
            [self.edim, self.edim],
            stddev,
            wline_asp
        )

        self.line_layer_ctx = LinearLayer_3dim(
            [self.edim, self.edim],
            stddev,
            wline_ctx
        )
        self.line_layer_output = LinearLayer_3dim(
            [self.edim, self.edim],
            stddev,
            wline_out
        )
        # version 2 start
        self.wline_ca = wline_att_ca or tf.Variable(
            tf.random_normal([self.edim, 1], stddev=stddev),
            trainable=True
        )
        # version 2 end

    def count_alpha(self, context, aspect,  output, ctx_bitmap, alpha_adj=None):
        '''
        Count the attention weights. 
        alpha = softmax(tanh(wa*asp + wb*ctx))

        Args:
            :type context: tensor, shape = [batch_size, time_steps, edim]
            :param context: the input context. 

            :type aspect: tensor, shape = [batch_size, edim]
            :param aspect: the input aspect. 

            :type ctx_bitmap: tensorflow, shape like context. 
            :param ctx_bitmap: the context's bitmap, use to remove the influence of padding. \

        Returns:
            A tensor.  The attention weights of the context. 
        '''
        time_steps = tf.shape(context)[1]
        aspect_3dim = tf.reshape(
            tf.tile(aspect, [1, time_steps]),
            [-1, time_steps, self.edim]
        )
        output_3dim = tf.reshape(
            tf.tile(output, [1, time_steps]),
            [-1, time_steps, self.edim]
        )

        res_asp = self.line_layer_asp.forward(aspect_3dim)
        res_ctx = self.line_layer_ctx.forward(context)
        res_output = self.line_layer_output.forward(output_3dim)

        res_sum = res_asp + res_ctx + res_output
        res_act = activer(res_sum, self.active)

        batch_size = tf.shape(context)[0]
        w_shp0 = tf.shape(self.wline_ca)[0]
        w_shp1 = tf.shape(self.wline_ca)[1]
        w_line_3dim = tf.reshape(
            tf.tile(self.wline_ca, [batch_size, 1]),
            [batch_size, w_shp0, w_shp1]
        )
        res_act = tf.reshape(
            tf.matmul(res_act, w_line_3dim),
            [-1, time_steps]
        )

        alpha = normalizer(self.norm_type ,res_act, ctx_bitmap, 1)
        if alpha_adj is not None:
            alpha += alpha_adj
        return alpha

    def count_alpha2(self, context, aspect,  output, ctx_bitmap, alpha_adj=None):
        '''
        Count the attention weights.
        alpha = softmax(tanh(wa*asp + wb*ctx))

        Args:
            :type context: tensor, shape = [batch_size, time_steps, edim]
            :param context: the input context.

            :type aspect: tensor, shape = [batch_size, edim]
            :param aspect: the input aspect.

            :type ctx_bitmap: tensorflow, shape like context.
            :param ctx_bitmap: the context's bitmap, use to remove the influence of padding. \

        Returns:
            A tensor.  The attention weights of the context.
        '''
        time_steps = tf.shape(context)[1]
        aspect_3dim = tf.reshape(
            tf.tile(aspect, [1, time_steps]),
            [-1, time_steps, self.edim]
        )

        res_asp = self.line_layer_asp.forward(aspect_3dim)
        res_ctx = self.line_layer_ctx.forward(context)
        res_output = self.line_layer_output.forward(output)

        res_sum = res_asp + res_ctx + res_output
        res_act = activer(res_sum, self.active)

        batch_size = tf.shape(context)[0]
        w_shp0 = tf.shape(self.wline_ca)[0]
        w_shp1 = tf.shape(self.wline_ca)[1]
        w_line_3dim = tf.reshape(
            tf.tile(self.wline_ca, [batch_size, 1]),
            [batch_size, w_shp0, w_shp1]
        )
        res_act = tf.reshape(
            tf.matmul(res_act, w_line_3dim),
            [-1, time_steps]
        )

        alpha = normalizer(self.norm_type ,res_act, ctx_bitmap, 1)
        if alpha_adj is not None:
            alpha += alpha_adj
        return alpha

    def forward(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        '''
        Weight sum the context,
        line transform aspect,
        add two of them. 

        Args:
            :type context: tensor
            :param context: the input context, shape = [batch_size, time_steps, edim]

            :type aspect: tensor
            :param aspect: the input aspect, shape = [batch_size, edim]
            
            :type output: tensor
            :param output: the last output, shape = [batch_size, edim]

            :type ctx_bitmap: tensor
            :param ctx_bitmap: the bitmap of context

        Returns:
            The sentence embedding. 
        '''
        mem_size = tf.shape(context)[1]
        context = context
        output = output
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha(
            context, aspect, output, ctx_bitmap, alpha_adj)
        # vec.shape = [batch_size, 1, edim]
        vec = tf.matmul(
            tf.reshape(alpha, [-1, 1, mem_size]),
            context
        )
        return vec, alpha

    def forward2(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        '''
        Weight sum the context,
        line transform aspect,
        add two of them.

        Args:
            :type context: tensor
            :param context: the input context, shape = [batch_size, time_steps, edim]

            :type aspect: tensor
            :param aspect: the input aspect, shape = [batch_size, edim]

            :type output: tensor
            :param output: the last output, shape = [batch_size, edim]

            :type ctx_bitmap: tensor
            :param ctx_bitmap: the bitmap of context

        Returns:
            The sentence embedding.
        '''
        mem_size = tf.shape(context)[1]
        context = context
        output = output
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha2(
            context, aspect, output, ctx_bitmap, alpha_adj)
        # vec.shape = [batch_size, 1, edim]
        vec = tf.matmul(
            tf.reshape(alpha, [-1, 1, mem_size]),
            context
        )
        return vec, alpha

    def forward_p(self, context, aspect, output, ctx_bitmap, location, alpha_adj=None):
        '''
        Weight sum the context,
        line transform aspect,
        add two of them.

        Args:
            :type context: tensor
            :param context: the input context, shape = [batch_size, time_steps, edim]

            :type aspect: tensor
            :param aspect: the input aspect, shape = [batch_size, edim]

            :type output: tensor
            :param output: the last output, shape = [batch_size, edim]

            :type ctx_bitmap: tensor
            :param ctx_bitmap: the bitmap of context

        Returns:
            The sentence embedding.
        '''
        mem_size = tf.shape(context)[1]
        context = context
        output = output
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha(
            context, aspect, output, ctx_bitmap, alpha_adj)
        # vec.shape = [batch_size, 1, edim]
        vec = tf.matmul(
            tf.add(tf.reshape(alpha, [-1, 1, mem_size]),location),
            context
        )
        return vec, alpha

    def forward_wot_sum(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        '''
        Weight sum the context,
        line transform aspect,
        add two of them. 

        Args:
            :type context: tensor
            :param context: the input context, shape = [batch_size, time_steps, edim]

            :type aspect: tensor
            :param aspect: the input aspect, shape = [batch_size, edim]

            :type ctx_bitmap: tensor
            :param ctx_bitmap: the bitmap of context

        Returns:
            The adjusted context base on the attention, has the same shape with the input. 
        '''
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        output = output
        # adjust attention, alpha.shape = [batch_size, time_step]
        alpha = self.count_alpha(
            context, aspect, output, ctx_bitmap, alpha_adj)
        alpha_3dim = tf.tile(
            tf.reshape(alpha, [-1, mem_size, 1]),
            [1, 1, self.edim]
        )
        ret = context * alpha_3dim
        return ret ,alpha
