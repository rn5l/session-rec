
import tensorflow as tf
import math
import numpy as np


def encoder(inputs, input_mask, length, hidden_size, num_heads=1, num_block=2, drop_out=0.4, train=True):
    with tf.variable_scope('encoder'):
        enc = pos_encoding(length, hidden_size)
        encoder_inputs = inputs + enc
        encoder_inputs = tf.layers.dropout(encoder_inputs, rate=drop_out, training=tf.convert_to_tensor(train))
        for i in range(num_block):
            with tf.variable_scope("num_blocks_{}".format(i)):
                enc = multihead_attention(encoder_inputs,
                                          input_mask,
                                          encoder_inputs,
                                          input_mask,
                                          num_units=hidden_size,
                                          num_heads=num_heads,
                                          dropout_rate=drop_out,
                                          is_training=train,
                                          causality=False)
                enc = feedforward(enc, num_units=[num_heads*hidden_size, hidden_size])
    return enc


def decoder(dec_inputs, dec_mask, dec_length, enc_inputs, enc_mask, hidden_size,
            num_heads=1, num_block=2, drop_out=0.4, train=True):
    with tf.variable_scope("decoder"):
        dec = pos_encoding(dec_length, hidden_size)
        decoder_inputs = dec_inputs + dec
        decoder_inputs = tf.layers.dropout(decoder_inputs, rate=drop_out, training=tf.convert_to_tensor(train))
        for i in range(num_block):
            with tf.variable_scope("num_blocks_{}".format(i)):
                dec = multihead_attention(decoder_inputs,
                                          dec_mask,
                                          decoder_inputs,
                                          dec_mask,
                                          num_units=hidden_size,
                                          num_heads=num_heads,
                                          dropout_rate=drop_out,
                                          is_training=train,
                                          causality=True,
                                          scope="self_attention")
                #dec = feedforward(dec, num_units=[num_heads * hidden_size, hidden_size])
                dec = multihead_attention(dec,
                                          dec_mask,
                                          enc_inputs,
                                          enc_mask,
                                          num_units=hidden_size,
                                          num_heads=num_heads,
                                          dropout_rate=drop_out,
                                          is_training=train,
                                          causality=False,
                                          scope="vanilla_attention")
                dec = feedforward(dec, num_units=[num_heads*hidden_size, hidden_size])
    return dec#tf.reduce_mean(dec, axis=1)


def multihead_attention(queries,
                        query_masks,
                        keys,
                        key_masks,
                        num_units=None,
                        num_heads=1,
                        dropout_rate=0.0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        ##---增加normalize用于last数据-------------------
        #queries = normalize(queries)
        #keys = normalize(keys)
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, use_bias=False, name='q')  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu, use_bias=False, name='k')  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu, use_bias=False, name='v')  # (N, T_k, C)
        #V = keys

        # Split and concat
        # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        # K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        # V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        #key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        #query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V)  # ( h*N, T_q, C/h)

        # Restore shape
        #outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[100, 100],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        #Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs



def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


# def pos_encoding(sentence_length, dim, dtype=tf.float32):
#
#     encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in tf.range(sentence_length) for i in range(dim)])
#     encoded_vec[::2] = np.sin(encoded_vec[::2])
#     encoded_vec[1::2] = np.cos(encoded_vec[1::2])
#
#     return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def pos_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


