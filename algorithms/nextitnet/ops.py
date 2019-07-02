import tensorflow as tf
import math



#config e.g. dilations: [1,4,16,] In most cases[1,4,] is enough
def nextitnet_residual_block(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True, key=''):
    resblock_type = "decoder"
    resblock_name = "{}nextitnet_residual_block{}_layer_{}_{}".format(key, resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name=key+"layer_norm1", trainable=train)
        #input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name=key+"dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name=key+"layer_norm2", trainable=train)
        #input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1

#suggest using this one if your data has strong sequence,  dilations: [1,2,4,1,2,4,]
#Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Identity mappings in deep residual networks.
def nextitnet_residual_block_one(input_, dilation, layer_id,
    residual_channels, kernel_size,
    causal = True, train = True, key=''):
        resblock_type = "decoder"
        resblock_name = "{}nextitnet_residual_block_one_{}_layer_{}_{}".format(key,resblock_type, layer_id, dilation)
        with tf.variable_scope(resblock_name):
            input_ln = layer_norm(input_, name="layer_norm1", trainable = train)
            relu1 = tf.nn.relu(input_ln)
            conv1 = conv1d(relu1, int(0.5*residual_channels), name = key+"conv1d_1")
            conv1 = layer_norm(conv1, name=key+"layer_norm2", trainable = train)
            relu2 = tf.nn.relu(conv1)
            
            dilated_conv = conv1d(relu2, int(0.5*residual_channels),
                dilation, kernel_size,
                causal = causal,
                name = "dilated_conv"
                )

            dilated_conv = layer_norm(dilated_conv, name=key+"layer_norm3", trainable = train)
            relu3 = tf.nn.relu(dilated_conv)
            conv2 = conv1d(relu3,  residual_channels, name = key+'conv1d_2')
            return input_ + conv2


#seems not good
#Conditional Image Generation with PixelCNN Decoders, wrong implementation?? let me know if you find the problem
def nextitnet_residual_block_gatedCNN(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True, key=''):
    resblock_type = "decoder"
    resblock_name = "{}gatedCNN_{}_layer_{}_{}".format(key,resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name):


        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name=key+"dilated_conv"
                              )
        tanh=tf.nn.tanh(dilated_conv)
        gate_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name=key+"gate_conv"
                              )

        sigm = tf.nn.sigmoid(gate_conv)
        multi=tf.multiply(tanh,sigm)
        multi=conv1d(multi, residual_channels, name=key+"conv1d_1")

        return input_ + multi



def conv1d(input_, output_channels,
           dilation=1, kernel_size=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'SAME') + bias
            out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias

        return tf.squeeze(out, [1])


# tf.contrib.layers.layer_norm
def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])],
                               initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [int(shape[-1])],
                                initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta

