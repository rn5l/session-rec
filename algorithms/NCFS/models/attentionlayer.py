from keras import backend as K
from keras.engine import Layer
from keras import initializers, regularizers, constraints


EPSILON = 1e-32


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        y = K.dot(x, K.expand_dims(kernel))
        return K.squeeze(y, axis=-1)
    else:
        return K.dot(x, kernel)


def isr(x, alpha=1.):
    return x / K.sqrt(1 + alpha * K.square(x))


class ISRAttentionLayer(Layer):
    def __init__(self, alpha=1., keepdims=False, **kwargs):
        self.alpha = alpha
        self.keepdims = keepdims
        super(ISRAttentionLayer, self).__init__(**kwargs)


class Attention(ISRAttentionLayer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, alpha=1., keepdims=False, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 2.0.6
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        super(Attention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if isinstance(mask, list):
                mask = mask[0]
            mask = K.any(mask, axis=-1, keepdims=self.keepdims)

        return mask

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = isr(eij, self.alpha)
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        a /= K.sum(a, axis=1, keepdims=True) + EPSILON

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1) + input_shape[2:] if self.keepdims else (input_shape[0],) + input_shape[2:]


class TwoLayerAttention(ISRAttentionLayer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 mid_units=None,
                 alpha=1.,
                 keepdims=False,
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.mid_units = mid_units
        self.supports_masking = True
        super(TwoLayerAttention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.mid_units is None:
            self.mid_units = input_shape[-1]

        self.wt_mid = self.add_weight(shape=(input_shape[-1], self.mid_units),
                                      name='wt_mid',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_mid = self.add_weight(shape=(self.mid_units,),
                                         initializer='zero',
                                         name='b_mid',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.wt_out = self.add_weight(shape=(self.mid_units,),
                                      name='wt_out',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_out = self.add_weight(shape=(1,),
                                         initializer='zero',
                                         name='b_out',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)

        super(TwoLayerAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1) + input_shape[2:] if self.keepdims else (input_shape[0],) + input_shape[2:]

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = K.any(mask, axis=-1, keepdims=True)
        return mask

    def call(self, inputs, mask=None):
        e = K.dot(inputs, self.wt_mid)
        if self.use_bias:
            e += self.b_mid
        e = K.tanh(e)
        e = dot_product(e, self.wt_out)
        if self.use_bias:
            e += self.b_out

        e = isr(e, self.alpha)
        wt = K.exp(e)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            wt *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number to the sum.
            wt /= K.sum(wt, axis=1, keepdims=True) + EPSILON
        else:
            wt /= K.sum(wt, axis=1, keepdims=True)
            # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = inputs * K.expand_dims(wt)
        return K.sum(weighted_input, axis=1, keepdims=self.keepdims)


class BatchAttention(ISRAttentionLayer):
    def __init__(self, alpha=1., keepdims=False, **kwargs):
        self.supports_masking = True
        super(BatchAttention, self).__init__(alpha, keepdims, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1) + input_shape[0][2:] if self.keepdims else (input_shape[0][0],) + input_shape[0][2:]

    def compute_mask(self, inputs, mask=None):
        retmask = mask
        if isinstance(mask, list):
            if mask[0] is not None:
                retmask = K.any(mask[0], axis=-1, keepdims=True)
            if mask[1] is not None:
                retmask &= mask[1]

        return retmask

    def call(self, inputs, mask=None):
        # e = K.batch_dot(inputs[0], K.permute_dimensions(inputs[1], (0, 2, 1)))
        # e = K.squeeze(e, axis=-1)
        e = inputs[0] * inputs[1]
        e = K.sum(e, axis=-1)
        e = isr(e, self.alpha)
        wt = K.exp(e)

        # apply mask after the exp. will be re-normalized next
        if mask is not None and mask[0] is not None:
            mask = mask[0] if mask[1] is None else mask[0] & mask[1]
            wt *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number to the sum.
            wt /= K.sum(wt, axis=1, keepdims=True) + EPSILON
        else:
            wt /= K.sum(wt, axis=1, keepdims=True)
            # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = inputs[0] * K.expand_dims(wt)
        return K.sum(weighted_input, axis=1, keepdims=self.keepdims)


class BiasedAttention(ISRAttentionLayer):
    def __init__(self, alpha=1., keepdims=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):
        self.supports_masking = True
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        super(BiasedAttention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[0][-1],),
                                      name='weight',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        super(BiasedAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1) + input_shape[0][2:] if self.keepdims else (input_shape[0][0],) + input_shape[0][2:]

    def compute_mask(self, inputs, mask=None):
        retmask = mask
        if isinstance(mask, list):
            if mask[0] is not None:
                retmask = K.any(mask[0], axis=-1, keepdims=True)
            if mask[1] is not None:
                retmask &= mask[1]

        return retmask

    def call(self, inputs, mask=None):
        e = inputs[0] * (inputs[1] + self.W)
        e = K.sum(e, axis=-1)
        e = isr(e, self.alpha)
        wt = K.exp(e)

        # apply mask after the exp. will be re-normalized next
        if mask is not None and mask[0] is not None:
            mask = mask[0] if mask[1] is None else mask[0] & mask[1]
            wt *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
            wt /= K.sum(wt, axis=1, keepdims=True) + EPSILON
        else:
            wt /= K.sum(wt, axis=1, keepdims=True)
            # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = inputs[0] * K.expand_dims(wt)
        return K.sum(weighted_input, axis=1, keepdims=self.keepdims)


class PairAttention(ISRAttentionLayer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 mid_units=None,
                 alpha=1.,
                 keepdims=False,
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.mid_units = mid_units
        self.supports_masking = True
        super(PairAttention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.mid_units is None:
            self.mid_units = input_shape[0][-1]

        self.wt_mid = self.add_weight(shape=(input_shape[0][-1] + input_shape[1][-1], self.mid_units),
                                      name='wt_mid',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_mid = self.add_weight(shape=(self.mid_units,),
                                         initializer='zero',
                                         name='b_mid',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.wt_out = self.add_weight(shape=(self.mid_units,),
                                      name='wt_out',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_out = self.add_weight(shape=(1,),
                                         initializer='zero',
                                         name='b_out',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)

        super(PairAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1) + input_shape[0][2:] if self.keepdims else (input_shape[0][0],) + input_shape[0][
                                                                                                        2:]

    def compute_mask(self, inputs, mask=None):
        retmask = mask
        if isinstance(mask, list):
            if mask[0] is not None:
                retmask = K.any(mask[0], axis=-1, keepdims=True)
            if mask[1] is not None:
                retmask &= mask[1]

        return retmask

    def call(self, inputs, mask=None):
        rep_input1 = K.repeat(K.squeeze(inputs[1], axis=1), inputs[0].shape[1]) if inputs[1].shape[1] == 1 else inputs[
            1]
        conca_input = K.concatenate([inputs[0], rep_input1])
        e = K.dot(conca_input, self.wt_mid)
        if self.use_bias:
            e += self.b_mid
        e = K.tanh(e)
        e = dot_product(e, self.wt_out)
        if self.use_bias:
            e += self.b_out

        e = isr(e, self.alpha)
        wt = K.exp(e)

        # apply mask after the exp. will be re-normalized next
        if mask is not None and mask[0] is not None:
            mask = mask[0] if mask[1] is None else mask[0] & mask[1]
            wt *= K.cast(mask, K.floatx())
            # in some cases especially in the early stages of training the sum may be almost zero
            # and this results in NaN's. A workaround is to add a very small positive number to the sum.
            wt /= K.sum(wt, axis=1, keepdims=True) + EPSILON
        else:
            wt /= K.sum(wt, axis=1, keepdims=True)
            # a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = inputs[0] * K.expand_dims(wt)
        return K.sum(weighted_input, axis=1, keepdims=self.keepdims)


class AverageAttention(ISRAttentionLayer):
    def __init__(self, alpha=1., keepdims=False, **kwargs):
        self.supports_masking = True
        self.keepdims = keepdims
        super(AverageAttention, self).__init__(alpha, keepdims, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1) + input_shape[2:] if self.keepdims else (input_shape[0],) + input_shape[2:]

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = K.any(mask, axis=-1, keepdims=self.keepdims)

        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask_float = K.cast(mask, K.floatx())
            mask_float /= (K.sum(mask_float, axis=1, keepdims=True) + EPSILON)
            inputs *= K.expand_dims(mask_float)
            return K.sum(inputs, axis=1, keepdims=self.keepdims)
        return K.mean(inputs, axis=1, keepdims=self.keepdims)


class GatedAttention(ISRAttentionLayer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 mid_units=64,
                 use_bias=True,
                 alpha=1.,
                 keepdims=False,
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.mid_units = mid_units
        self.use_bias = use_bias
        self.alpha = alpha
        self.supports_masking = True
        super(GatedAttention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.mid_units is None:
            self.mid_units = input_shape[0][-1]

        self.wt_mid = self.add_weight(shape=(input_shape[0][-1] + input_shape[1][-1], self.mid_units),
                                      name='wt_mid',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_mid = self.add_weight(shape=(self.mid_units,),
                                         initializer='zero',
                                         name='b_mid',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.wt_out = self.add_weight(shape=(self.mid_units,),
                                      name='wt_out',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_out = self.add_weight(shape=(1,),
                                         initializer='zero',
                                         name='b_out',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)

        super(GatedAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1) + input_shape[0][2:] if self.keepdims else (input_shape[0][0],) + input_shape[0][2:]

    def compute_mask(self, inputs, mask=None):
        retmask = mask
        if isinstance(mask, list):
            if mask[0] is not None:
                retmask = K.any(mask[0], axis=-1, keepdims=True)
            if mask[1] is not None:
                retmask &= mask[1]

        return retmask

    def call(self, inputs, mask=None):
        e = K.dot(K.concatenate([inputs[0], inputs[1]], axis=-1), self.wt_mid)
        if self.use_bias:
            e += self.b_mid
        e = K.tanh(e)
        e = dot_product(e, self.wt_out)
        if self.use_bias:
            e += self.b_out

        e = isr(e, self.alpha)
        update_gate = K.sigmoid(e)
        update_gate_compl = 1 - update_gate

        if mask[0] is not None:
            update_gate_compl *= K.cast(mask[0], K.floatx())

        if mask[1] is not None:
            update_gate *= K.cast(mask[1], K.floatx())

        sum_gate = update_gate + update_gate_compl + EPSILON
        update_gate = K.expand_dims(update_gate / sum_gate)
        update_gate_compl = K.expand_dims(update_gate_compl / sum_gate)

        output = update_gate_compl * inputs[0] + update_gate * inputs[1]
        if not self.keepdims: output = K.squeeze(output, axis=1)

        return output

class AggGatedAttention(ISRAttentionLayer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 mid_units=64,
                 alpha=1e-3,
                 normalize = True,
                 keepdims=False,
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.mid_units = mid_units
        self.alpha = alpha
        self.normalize = normalize
        self.supports_masking = True
        super(AggGatedAttention, self).__init__(alpha, keepdims, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.mid_units is None:
            self.mid_units = input_shape[0][-1]

        self.wt_mid = self.add_weight(shape=(input_shape[0][-1] + input_shape[1][-1], self.mid_units),
                                      name='wt_mid',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_mid = self.add_weight(shape=(self.mid_units,),
                                         initializer='zero',
                                         name='b_mid',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)
        self.wt_out = self.add_weight(shape=(self.mid_units,),
                                      name='wt_out',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.b_out = self.add_weight(shape=(1,),
                                         initializer='zero',
                                         name='b_out',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint)

        super(AggGatedAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1) + input_shape[0][2:] if self.keepdims else (input_shape[0][0],) + input_shape[0][2:]

    def compute_mask(self, inputs, mask=None):
        retmask = mask
        if isinstance(mask, list):
            if mask[0] is not None:
                retmask = K.any(mask[0], axis=-1, keepdims=True)
            if mask[1] is not None:
                retmask &= mask[1]

        return retmask

    def call(self, inputs, mask=None):
        e = K.dot(K.concatenate([inputs[0], inputs[1]], axis=-1), self.wt_mid)
        if self.use_bias:
            e += self.b_mid
        e = K.tanh(e)
        e = dot_product(e, self.wt_out)
        if self.use_bias:
            e += self.b_out

        e = isr(e, self.alpha)
        update_gate = K.sigmoid(e)

        # apply mask after the exp. will be re-normalized next
        if mask[1] is not None:
            update_gate *= K.cast(mask[1], K.floatx())

        update_gate = K.expand_dims(update_gate)

        if self.normalize:
            update_gate /= 1 + update_gate
            output = (1 - update_gate) * inputs[0] + update_gate * inputs[1]
        else:
            output = inputs[0] + update_gate * inputs[1]

        if not self.keepdims: output = K.squeeze(output, axis=1)

        return output
