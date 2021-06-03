# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""

import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import pandas as pd
from collections import OrderedDict
import logging
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

srng = RandomStreams()


def inspect(tvar):
    return tvar.get_value(borrow=True)


def print_norm(tvar, name='var'):
    logger.info('{}: {:.4f}'.format(name, np.linalg.norm(inspect(tvar))))


class Sampler:
    def __init__(self, data, n_sample, rng=None, item_key='item_id', sample_alpha=0.75, sample_store=10000000):
        self.sample_alpha = sample_alpha
        self.sample_store = sample_store
        self.n_sample = n_sample
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng

        self.pop = data[item_key].value_counts() ** sample_alpha
        self.pop = self.pop.cumsum() / self.pop.sum()
        if self.sample_store:
            self.generate_length = self.sample_store // self.n_sample
            if self.generate_length <= 1:
                self.sample_store = 0
                logger.info('No example store was used')
            else:
                self.neg_samples = self._generate_neg_samples(self.pop, self.generate_length)
                self.sample_pointer = 0
                logger.info('Created sample store with {} batches of samples'.format(self.generate_length))
        else:
            logger.info('No example store was used')

    def next_sample(self):
        if self.sample_store:
            if self.sample_pointer == self.generate_length:
                self.neg_samples = self._generate_neg_samples(self.pop, self.generate_length)
                self.sample_pointer = 0
            sample = self.neg_samples[self.sample_pointer]
            self.sample_pointer += 1
        else:
            sample = self._generate_neg_samples(self.pop, 1)
        return sample

    def _generate_neg_samples(self, pop, length):
        n_items = pop.shape[0]
        if self.sample_alpha:
            sample = np.searchsorted(pop, self.rng.rand(self.n_sample * length))
        else:
            sample = self.rng.choice(n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample


class HGRU4Rec:
    """
    HGRU4Rec(session_layers, user_layers, n_epochs=10, batch_size=50,
             learning_rate=0.05, momentum=0.0,
             adapt='adagrad', decay=0.9, grad_cap=0, sigma=0,
             dropout_p_hidden_usr=0.0,
             dropout_p_hidden_ses=0.0, dropout_p_init=0.0,
             init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None,
             train_random_order=False, lmbd=0.0,
             session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId', n_sample=0,
             sample_alpha=0.75,
             item_embedding=None, init_item_embeddings=None,
             user_hidden_bias_mode='init', user_output_bias=False,
             user_to_session_act='tanh', seed=42)
    Initializes the network.

    Parameters
    -----------
    session_layers : 1D array
        list of the number of GRU units in the session layers
    user_layers : 1D array
        list of the number of GRU units in the user layers
    n_epochs : int
        number of training epochs (default: 10)
    batch_size : int
        size of the minibatch, also effect the number of negative samples through minibatch based sampling (default: 50)
    dropout_p_hidden_usr : float
        probability of dropout of hidden units for the user layers (default: 0.0)
    dropout_p_hidden_ses : float
        probability of dropout of hidden units for the session layers (default: 0.0)
    dropout_p_init : float
        probability of dropout of the session-level initialization (default: 0.0)
    learning_rate : float
        learning rate (default: 0.05)
    momentum : float
        if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the appropriate learning rate adaptation strategy, use None for standard SGD (default: 'adagrad')
    decay : float
        decay parameter for RMSProp, has no effect in other modes (default: 0.9)
    grad_cap : float
        clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
    init_as_normal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    reset_after_session : boolean
        whether the hidden state is set to zero after a session finished (default: True)
    loss : 'top1', 'bpr' or 'cross-entropy'
        selects the loss function (default: 'top1')
    hidden_act : 'tanh' or 'relu'
        selects the activation function on the hidden states (default: 'tanh')
    final_act : None, 'linear', 'relu' or 'tanh'
        selects the activation function of the final layer where appropriate, None means default (tanh if the loss is brp or top1; softmax for cross-entropy),
        cross-entropy is only affeted by 'tanh' where the softmax layers is preceeded by a tanh nonlinearity (default: None)
    train_random_order : boolean
        whether to randomize the order of sessions in each epoch (default: False)
    lmbd : float
        coefficient of the L2 regularization (default: 0.0)
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    user_key : string
        header of the user column in the input file (default: 'UserId')
    n_sample : int
        number of additional negative samples to be used (besides the other examples of the minibatch) (default: 0)
    sample_alpha : float
        the probability of an item used as an additional negative sample is supp^sample_alpha (default: 0.75)
        (e.g.: sample_alpha=1 --> popularity based sampling; sample_alpha=0 --> uniform sampling)
    item_embedding: int
        size of the item embedding vector (default: None)
    init_item_embeddings: 2D array or dict
        array with the initial values of the embeddings vector of every item,
        or dict that maps each item id to its embedding vector (default: None)
    user_propagation_mode: string
        'init' to use the (last) user hidden state to initialize the (first) session hidden state;
        'all' to propagate the user hidden also in input the the (first) session layers. (default: 'init')
    user_to_output: boolean
        True to propagate the (last) user hidden state in input to the final output layer, False otherwise (default: False)
    user_to_session_act: string
        activation of the user-to-session initialization network (default: 'tanh')
    seed: int
        random seed (default: 42)
    """

    def __init__(self, session_layers, user_layers, n_epochs=10, batch_size=50, learning_rate=0.05, momentum=0.0,
                 adapt='adagrad', decay=0.9, grad_cap=0, sigma=0, dropout_p_hidden_usr=0.0,
                 dropout_p_hidden_ses=0.0, dropout_p_init=0.0, init_as_normal=False,
                 reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False,
                 lmbd=0.0, session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId', n_sample=0,
                 sample_alpha=0.75, item_embedding=None, init_item_embeddings=None, user_propagation_mode='init',
                 user_to_output=False, user_to_session_act='tanh', seed=42):
        self.session_layers = session_layers
        self.user_layers = user_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden_usr = dropout_p_hidden_usr
        self.dropout_p_hidden_ses = dropout_p_hidden_ses
        self.dropout_p_init = dropout_p_init
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.reset_after_session = reset_after_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key
        self.grad_cap = grad_cap
        self.train_random_order = train_random_order
        self.lmbd = lmbd

        self.user_propagation_mode = user_propagation_mode
        self.user_to_output = user_to_output

        self.item_embedding = item_embedding
        self.init_item_embeddings = init_item_embeddings

        self.rng = np.random.RandomState(seed=seed)

        if adapt == 'rmsprop':
            self.adapt = 'rmsprop'
        elif adapt == 'adagrad':
            self.adapt = 'adagrad'
        elif adapt == 'adadelta':
            self.adapt = 'adadelta'
        elif adapt == 'adam':
            self.adapt = 'adam'
        else:
            self.adapt = False
        if loss == 'cross-entropy':
            if final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif loss == 'bpr':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif loss == 'top1':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError('loss {} not implemented'.format(loss))
        if hidden_act == 'relu':
            self.hidden_activation = self.relu
        elif hidden_act == 'tanh':
            self.hidden_activation = self.tanh
        else:
            raise NotImplementedError('hidden activation {} not implemented'.format(hidden_act))
        if user_to_session_act == 'relu':
            self.s_init_act = self.relu
        elif user_to_session_act == 'tanh':
            self.s_init_act = self.tanh
        else:
            raise NotImplementedError('user-to-session activation {} not implemented'.format(hidden_act))

        self.n_sample = n_sample
        self.sample_alpha = sample_alpha

    ######################ACTIVATION FUNCTIONS#####################
    def linear(self, X):
        return X

    def tanh(self, X):
        return T.tanh(X)

    def softmax(self, X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def softmaxth(self, X):
        X = self.tanh(X)
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def relu(self, X):
        return T.maximum(X, 0)

    def sigmoid(self, X):
        return T.nnet.sigmoid(X)

    #################################LOSS FUNCTIONS################################
    def cross_entropy(self, yhat):
        return T.cast(T.mean(-T.log(T.diag(yhat) + 1e-24)), theano.config.floatX)

    def bpr(self, yhat):
        return T.cast(T.mean(-T.log(T.nnet.sigmoid(T.diag(yhat) - yhat.T))), theano.config.floatX)

    def top1(self, yhat):
        yhatT = yhat.T
        return T.cast(T.mean(
            T.mean(T.nnet.sigmoid(-T.diag(yhat) + yhatT) + T.nnet.sigmoid(yhatT ** 2), axis=0) - T.nnet.sigmoid(
                T.diag(yhat) ** 2) / self.batch_size), theano.config.floatX)

    ###############################################################################
    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)

    def init_weights(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return theano.shared(self.floatX(self.rng.randn(*shape) * sigma), borrow=True)
        else:
            return theano.shared(self.floatX(self.rng.rand(*shape) * sigma * 2 - sigma), borrow=True)

    def init_matrix(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return self.floatX(self.rng.randn(*shape) * sigma)
        else:
            return self.floatX(self.rng.rand(*shape) * sigma * 2 - sigma)

    def extend_weights(self, W, n_new):
        matrix = W.get_value()
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (matrix.shape[0] + matrix.shape[1] + n_new))
        if self.init_as_normal:
            new_rows = self.floatX(self.rng.randn(n_new, matrix.shape[1]) * sigma)
        else:
            new_rows = self.floatX(self.rng.rand(n_new, matrix.shape[1]) * sigma * 2 - sigma)
        W.set_value(np.vstack([matrix, new_rows]))

    def set_item_embeddings(self, E, values):
        if isinstance(values, dict):
            keys, values = values.keys(), np.vstack(list(values.values()))
        elif isinstance(values, np.ndarray):
            # use item ids ranging from 0 to the number of rows in values
            keys, values = np.arange(values.shape[0]), values
        else:
            raise NotImplementedError('Unsupported type')
        # map item ids to the internal indices
        mask = np.in1d(keys, self.itemidmap.index, assume_unique=True)
        idx = self.itemidmap[keys].dropna().values.astype(np.int)
        emb = E.get_value()
        emb[idx] = values[mask]
        E.set_value(emb)

    def preprocess_data(self, data):
        # sort by user and time key in order
        data.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
        data.reset_index(drop=True, inplace=True)
        offset_session = np.r_[0, data.groupby([self.user_key, self.session_key], sort=False).size().cumsum()[:-1]]
        user_indptr = np.r_[0, data.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum()[:-1]]
        return user_indptr, offset_session

    def save_state(self):
        state = OrderedDict()
        for i in range(len(self.session_layers)):
            state['Ws_in_' + str(i)] = self.Ws_in[i].get_value()
            state['Ws_hh_' + str(i)] = self.Ws_hh[i].get_value()
            state['Ws_rz_' + str(i)] = self.Ws_rz[i].get_value()
            state['Bs_h_' + str(i)] = self.Bs_h[i].get_value()
            state['Hs_' + str(i)] = self.Hs[i].get_value()
        state['Wsy'] = self.Wsy.get_value()
        state['By'] = self.By.get_value()
        for i in range(len(self.user_layers)):
            state['Wu_in_' + str(i)] = self.Wu_in[i].get_value()
            state['Wu_hh_' + str(i)] = self.Wu_hh[i].get_value()
            state['Wu_rz_' + str(i)] = self.Wu_rz[i].get_value()
            state['Bu_h_' + str(i)] = self.Bu_h[i].get_value()
            state['Hu_' + str(i)] = self.Hu[i].get_value()
        if self.user_to_output:
            state['Wuy'] = self.Wuy.get_value()
        state['Wu_to_s_init'] = self.Ws_init[0].get_value()
        state['Bu_to_s_init'] = self.Bs_init[0].get_value()
        if self.user_propagation_mode == 'all':
            state['Wu_to_s'] = self.Wu_to_s[0].get_value()
        return state

    def load_state(self, state):
        for i in range(len(self.session_layers)):
            self.Ws_in[i].set_value(state['Ws_in_' + str(i)], borrow=True)
            self.Ws_hh[i].set_value(state['Ws_hh_' + str(i)], borrow=True)
            self.Ws_rz[i].set_value(state['Ws_rz_' + str(i)], borrow=True)
            self.Bs_h[i].set_value(state['Bs_h_' + str(i)], borrow=True)
            self.Hs[i].set_value(state['Hs_' + str(i)], borrow=True)
        self.Wsy.set_value(state['Wsy'], borrow=True)
        self.By.set_value(state['By'], borrow=True)
        for i in range(len(self.user_layers)):
            self.Wu_in[i].set_value(state['Wu_in_' + str(i)], borrow=True)
            self.Wu_hh[i].set_value(state['Wu_hh_' + str(i)], borrow=True)
            self.Wu_rz[i].set_value(state['Wu_rz_' + str(i)], borrow=True)
            self.Bu_h[i].set_value(state['Bu_h_' + str(i)], borrow=True)
            self.Hu[i].set_value(state['Hu_' + str(i)], borrow=True)
        if self.user_to_output:
            self.Wuy.set_value(state['Wuy'], borrow=True)
        self.Ws_init[0].set_value(state['Wu_to_s_init'], borrow=True)
        self.Bs_init[0].set_value(state['Bu_to_s_init'], borrow=True)
        if self.user_propagation_mode == 'all':
            self.Wu_to_s[0].set_value(state['Wu_to_s'], borrow=True)

    def print_state(self):
        for i in range(len(self.session_layers)):
            print_norm(self.Ws_in[i], 'Ws_in_' + str(i))
            print_norm(self.Ws_hh[i], 'Ws_hh_' + str(i))
            print_norm(self.Ws_rz[i], 'Ws_rz_' + str(i))
            print_norm(self.Bs_h[i], 'Bs_h_' + str(i))
            print_norm(self.Hs[i], 'Hs_' + str(i))
        print_norm(self.Wsy, 'Wsy')
        print_norm(self.By, 'By')
        for i in range(len(self.user_layers)):
            print_norm(self.Wu_in[i], 'Wu_in_' + str(i))
            print_norm(self.Wu_hh[i], 'Wu_hh_' + str(i))
            print_norm(self.Wu_rz[i], 'Wu_rz_' + str(i))
            print_norm(self.Bu_h[i], 'Bu_h_' + str(i))
            print_norm(self.Hu[i], 'Hu_' + str(i))
        if self.user_to_output:
            print_norm(self.Wuy, 'Wuy')
        print_norm(self.Ws_init[0], 'Wu_to_s_init')
        print_norm(self.Bs_init[0], 'Bu_to_s_init')
        if self.user_propagation_mode == 'all':
            print_norm(self.Wu_to_s[0], 'Wu_to_s')

    def init_params(self):

        self.session_layers = [int(x) for x in str(self.session_layers).split(',')]
        self.user_layers = [int(x) for x in str(self.user_layers).split(',')]

        # TODO: I have not handle these lines, which was before creating the model in the original code
        # item_embedding_values = None
        # if args.load_item_embeddings is not None:
        #     item_embedding_values = np.load(args.load_item_embeddings)


        rnn_input_size = self.n_items
        if self.item_embedding is not None:
            self.E_item = self.init_weights((self.n_items, self.item_embedding))
            if self.init_item_embeddings is not None:
                self.set_item_embeddings(self.E_item, self.init_item_embeddings)
            rnn_input_size = self.item_embedding

        # Initialize the session parameters
        self.Ws_in, self.Ws_hh, self.Ws_rz, self.Bs_h, self.Hs = [], [], [], [], []
        for i in range(len(self.session_layers)):
            m = []
            m.append(
                self.init_matrix((self.session_layers[i - 1] if i > 0 else rnn_input_size, self.session_layers[i])))
            m.append(
                self.init_matrix((self.session_layers[i - 1] if i > 0 else rnn_input_size, self.session_layers[i])))
            m.append(
                self.init_matrix((self.session_layers[i - 1] if i > 0 else rnn_input_size, self.session_layers[i])))
            self.Ws_in.append(theano.shared(value=np.hstack(m), borrow=True))
            self.Ws_hh.append(self.init_weights((self.session_layers[i], self.session_layers[i])))
            m2 = []
            m2.append(self.init_matrix((self.session_layers[i], self.session_layers[i])))
            m2.append(self.init_matrix((self.session_layers[i], self.session_layers[i])))
            self.Ws_rz.append(theano.shared(value=np.hstack(m2), borrow=True))
            self.Bs_h.append(
                theano.shared(value=np.zeros((self.session_layers[i] * 3,), dtype=theano.config.floatX), borrow=True))
            self.Hs.append(
                theano.shared(value=np.zeros((self.batch_size, self.session_layers[i]), dtype=theano.config.floatX),
                              borrow=True))
        # Session to output weights
        self.Wsy = self.init_weights((self.n_items, self.session_layers[-1]))
        # Global output bias
        self.By = theano.shared(value=np.zeros((self.n_items, 1), dtype=theano.config.floatX), borrow=True)

        # Initialize the user parameters
        self.Wu_in, self.Wu_hh, self.Wu_rz, self.Bu_h, self.Hu = [], [], [], [], []
        for i in range(len(self.user_layers)):
            m = []
            m.append(self.init_matrix(
                (self.user_layers[i - 1] if i > 0 else self.session_layers[-1], self.user_layers[i])))
            m.append(self.init_matrix(
                (self.user_layers[i - 1] if i > 0 else self.session_layers[-1], self.user_layers[i])))
            m.append(self.init_matrix(
                (self.user_layers[i - 1] if i > 0 else self.session_layers[-1], self.user_layers[i])))
            self.Wu_in.append(theano.shared(value=np.hstack(m), borrow=True))
            self.Wu_hh.append(self.init_weights((self.user_layers[i], self.user_layers[i])))
            m2 = []
            m2.append(self.init_matrix((self.user_layers[i], self.user_layers[i])))
            m2.append(self.init_matrix((self.user_layers[i], self.user_layers[i])))
            self.Wu_rz.append(theano.shared(value=np.hstack(m2), borrow=True))
            self.Bu_h.append(
                theano.shared(value=np.zeros((self.user_layers[i] * 3,), dtype=theano.config.floatX), borrow=True))
            self.Hu.append(
                theano.shared(value=np.zeros((self.batch_size, self.user_layers[i]), dtype=theano.config.floatX),
                              borrow=True))
        if self.user_to_output:
            # User to output weights
            self.Wuy = self.init_weights((self.n_items, self.user_layers[-1]))

        # User-to-Session parameters
        self.Ws_init, self.Bs_init = [], []
        self.Ws_init.append(self.init_weights((self.user_layers[-1], self.session_layers[0])))
        self.Bs_init.append(
            theano.shared(value=np.zeros((self.session_layers[0],), dtype=theano.config.floatX), borrow=True))
        if self.user_propagation_mode == 'all':
            m = []
            m.append(self.init_matrix((self.user_layers[-1], self.session_layers[0])))
            m.append(self.init_matrix((self.user_layers[-1], self.session_layers[0])))
            m.append(self.init_matrix((self.user_layers[-1], self.session_layers[0])))
            self.Wu_to_s = [theano.shared(value=np.hstack(m), borrow=True)]

    def dropout(self, X, drop_p):
        if drop_p > 0:
            retain_prob = 1 - drop_p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        return X

    def adam(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        meang = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        countt = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            meang_new = v1 * meang + v2 * grad
            countt_new = countt + 1
            updates[acc] = acc_new
            updates[meang] = meang_new
            updates[countt] = countt_new
        else:
            acc_s = acc[sample_idx]
            meang_s = meang[sample_idx]
            countt_s = countt[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            meang_new = v1 * meang_s + v2 * grad
            countt_new = countt_s + 1.0
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            updates[meang] = T.set_subtensor(meang_s, meang_new)
            updates[countt] = T.set_subtensor(countt_s, countt_new)
        return (meang_new / (1 - v1 ** countt_new)) / (T.sqrt(acc_new / (1 - v1 ** countt_new)) + epsilon)

    def adagrad(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling

    def adadelta(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        upd = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
            grad = T.sqrt(upd + epsilon) * grad
            upd_new = v1 * upd + v2 * grad ** 2
            updates[upd] = upd_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            upd_s = upd[sample_idx]
            upd_new = v1 * upd_s + v2 * grad ** 2
            updates[upd] = T.set_subtensor(upd_s, upd_new)
            grad = T.sqrt(upd_s + epsilon) * grad
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling

    def rmsprop(self, param, grad, updates, sample_idx=None, epsilon=1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling

    def RMSprop(self, cost, params, full_params, sampled_params, sidxs, epsilon=1e-6):
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        sgrads = [T.grad(cost=cost, wrt=sparam) for sparam in sampled_params]
        updates = OrderedDict()
        if self.grad_cap > 0:
            norm = T.cast(T.sqrt(T.sum([T.sum([T.sum(g ** 2) for g in g_list]) for g_list in grads]) + T.sum(
                [T.sum(g ** 2) for g in sgrads])), theano.config.floatX)
            grads = [[T.switch(T.ge(norm, self.grad_cap), g * self.grad_cap / norm, g) for g in g_list] for g_list in
                     grads]
            sgrads = [T.switch(T.ge(norm, self.grad_cap), g * self.grad_cap / norm, g) for g in sgrads]
        for p_list, g_list in zip(params, grads):
            for p, g in zip(p_list, g_list):
                if self.adapt:
                    if self.adapt == 'adagrad':
                        g = self.adagrad(p, g, updates)
                    if self.adapt == 'rmsprop':
                        g = self.rmsprop(p, g, updates)
                    if self.adapt == 'adadelta':
                        g = self.adadelta(p, g, updates)
                    if self.adapt == 'adam':
                        g = self.adam(p, g, updates)
                if self.momentum > 0:
                    velocity = theano.shared(p.get_value(borrow=False) * 0., borrow=True)
                    velocity2 = self.momentum * velocity - np.float32(self.learning_rate) * (g + self.lmbd * p)
                    updates[velocity] = velocity2
                    updates[p] = p + velocity2
                else:
                    updates[p] = p * np.float32(1.0 - self.learning_rate * self.lmbd) - np.float32(
                        self.learning_rate) * g
        for i in range(len(sgrads)):
            g = sgrads[i]
            fullP = full_params[i]
            sample_idx = sidxs[i]
            sparam = sampled_params[i]
            if self.adapt:
                if self.adapt == 'adagrad':
                    g = self.adagrad(fullP, g, updates, sample_idx)
                if self.adapt == 'rmsprop':
                    g = self.rmsprop(fullP, g, updates, sample_idx)
                if self.adapt == 'adadelta':
                    g = self.adadelta(fullP, g, updates, sample_idx)
                if self.adapt == 'adam':
                    g = self.adam(fullP, g, updates, sample_idx)
            if self.lmbd > 0:
                delta = np.float32(self.learning_rate) * (g + self.lmbd * sparam)
            else:
                delta = np.float32(self.learning_rate) * g
            if self.momentum > 0:
                velocity = theano.shared(fullP.get_value(borrow=False) * 0., borrow=True)
                vs = velocity[sample_idx]
                velocity2 = self.momentum * vs - delta
                updates[velocity] = T.set_subtensor(vs, velocity2)
                updates[fullP] = T.inc_subtensor(sparam, velocity2)
            else:
                updates[fullP] = T.inc_subtensor(sparam, - delta)
        return updates

    def model(self, X, Sstart, Ustart, Hs, Hu, Y=None,
              drop_p_hidden_usr=0.0,
              drop_p_hidden_ses=0.0,
              drop_p_init=0.0):
        #
        # USER GRU
        #
        # update the User GRU with the last hidden state of the Session GRU
        # NOTE: the User GRU gets actually updated only when a new session starts
        user_in = T.dot(Hs[-1], self.Wu_in[0]) + self.Bu_h[0]
        user_in = user_in.T
        # ^ 3 * user_layers[0] x batch_size

        rz_u = T.nnet.sigmoid(user_in[self.user_layers[0]:]
                              + T.dot(Hu[0], self.Wu_rz[0]).T)
        # ^ 2 * user_layers[0] x batch_size

        h_u = self.hidden_activation(T.dot(Hu[0] * rz_u[:self.user_layers[0]].T, self.Wu_hh[0]).T
                                     + user_in[:self.user_layers[0]])
        # ^ user_layers[0] x batch_size

        z = rz_u[self.user_layers[0]:].T
        # batch_size x user_layers[0]
        h_u = (1.0 - z) * Hu[0] + z * h_u.T
        h_u = self.dropout(h_u, drop_p_hidden_usr)
        # ^ batch_size x user_layers[0]

        # update the User GRU only when a new session starts
        # Hu contains the state of the previous session
        h_u = Hu[0] * (1 - Sstart[:, None]) + h_u * Sstart[:, None]
        # ^ batch_size x user_layers[0]

        # reset the user network state for new users
        h_u = T.zeros_like(h_u) * Ustart[:, None] + h_u * (1 - Ustart[:, None])

        Hu_new = [h_u]
        for i in range(1, len(self.user_layers)):
            user_in = T.dot(h_u, self.Wu_in[i]) + self.Bu_h[i]
            user_in = user_in.T
            rz_u = T.nnet.sigmoid(user_in[self.user_layers[i]:]
                                  + T.dot(Hu[i], self.Wu_rz[i]).T)

            h_u = self.hidden_activation(T.dot(Hu[i] * rz_u[:self.user_layers[i]].T, self.Wu_hh[i]).T
                                         + user_in[:self.user_layers[i]])

            z = rz_u[self.user_layers[i]:].T
            h_u = (1.0 - z) * Hu[i] + z * h_u.T
            h_u = self.dropout(h_u, drop_p_hidden_usr)
            h_u = Hu[i] * (1 - Sstart[:, None]) + h_u * Sstart[:, None]
            h_u = T.zeros_like(h_u) * Ustart[:, None] + h_u * (1 - Ustart[:, None])
            Hu_new.append(h_u)

        #
        # SESSION GRU
        #
        # Process the input items
        if self.item_embedding is not None:
            # get the item embedding
            SE_item = self.E_item[X]  # sampled item embedding
            vec = T.dot(SE_item, self.Ws_in[0]) + self.Bs_h[0]
            Sin = SE_item
        else:
            Sx = self.Ws_in[0][X]
            vec = Sx + self.Bs_h[0]
            Sin = Sx
        session_in = vec.T
        # ^ session_layers[0] x batch_size

        # initialize the h_s with h_u only for starting sessions
        h_s_init = self.dropout(self.s_init_act(T.dot(h_u, self.Ws_init[0]) + self.Bs_init), drop_p_init)
        h_s = Hs[0] * (1 - Sstart[:, None]) + h_s_init * Sstart[:, None]
        # reset h_s for starting users
        h_s = h_s * (1 - Ustart[:, None]) + T.zeros_like(h_s) * Ustart[:, None]

        if self.user_propagation_mode == 'all':
            # this propagates the bias throughout all the session
            user_bias = T.dot(h_u, self.Wu_to_s[0]).T
            # ^ 3*session_layers[0] x batch_size

            # update the Session GRU
            rz_s = T.nnet.sigmoid(user_bias[self.session_layers[0]:]
                                  + session_in[self.session_layers[0]:]
                                  + T.dot(h_s, self.Ws_rz[0]).T)
            # ^ 2*session_layers[0] x batch_size

            h_s_cand = self.hidden_activation(T.dot(h_s * rz_s[:self.session_layers[0]].T, self.Ws_hh[0]).T
                                         + session_in[:self.session_layers[0]])
            # ^ session_layers[0] x batch_size
        else:
            rz_s = T.nnet.sigmoid(session_in[self.session_layers[0]:]
                                  + T.dot(h_s, self.Ws_rz[0]).T)
            h_s_cand = self.hidden_activation(T.dot(h_s * rz_s[:self.session_layers[0]].T, self.Ws_hh[0]).T
                                         + session_in[:self.session_layers[0]])

        z = rz_s[self.session_layers[0]:].T
        # ^ batch_size x session_layers[0]
        h_s = (1.0 - z) * h_s + z * h_s_cand.T
        h_s = self.dropout(h_s, drop_p_hidden_ses)
        # ^ batch_size x session_layers[0]
        Hs_new = [h_s]
        for i in range(1, len(self.session_layers)):
            # reset Hs for new starting users
            h_s_i = Hs[i] * (1 - Ustart[:, None]) + T.zeros_like(Hs[i]) * Ustart[:, None]
            # go through the next GRU layer
            session_in = T.dot(h_s, self.Ws_in[i]) + self.Bs_h[i]
            session_in = session_in.T
            rz_s = T.nnet.sigmoid(session_in[self.session_layers[i]:] + T.dot(h_s_i, self.Ws_rz[i]).T)
            h_s_i_cand = self.hidden_activation(T.dot(h_s_i * rz_s[:self.session_layers[i]].T, self.Ws_hh[i]).T
                                         + session_in[:self.session_layers[i]])
            z = rz_s[self.session_layers[i]:].T
            h_s_i = (1.0 - z) * h_s_i + z * h_s_i_cand.T
            h_s_i = self.dropout(h_s_i, drop_p_hidden_ses)
            Hs_new.append(h_s_i)

        if Y is not None:
            Ssy = self.Wsy[Y]
            SBy = self.By[Y]
            preact = T.dot(h_s, Ssy.T) + SBy.flatten()
            sampled_params = [Sin, Ssy, SBy]
            if self.user_to_output:
                Scy = self.Wuy[Y]
                preact += T.dot(h_u, Scy.T)
                sampled_params.append(Scy)
            y = self.final_activation(preact)
            return Hs_new, Hu_new, y, sampled_params
        else:
            preact = T.dot(h_s, self.Wsy.T) + self.By.flatten()
            if self.user_to_output:
                preact += T.dot(h_u, self.Wuy.T)
            y = self.final_activation(preact)
            return Hs_new, Hu_new, y, [Sin]

    def fit(self, train_data, test_data=None, valid_data=None, retrain=False, sample_store=10000000, patience=3, margin=1.003,
            save_to=None, load_from=None):
        '''
        Trains the network.

        Parameters
        --------
        train_data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        valid_data: pandas.DataFrame
            Validation data. If not none, it enables early stopping.
             Contains the transactions in the same format as in train_data, and it is used exclusively to compute the loss after each training iteration over train_data.
        retrain : boolean
            If False, do normal train. If True, do additional train (weights from previous trainings are kept as the initial network) (default: False)
        sample_store : int
            If additional negative samples are used (n_sample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).
        patience: int
            Patience of the early stopping procedure. Number of iterations with not decreasing validation loss before terminating the training procedure
        margin: float
            Margin of early stopping. Percentage improvement over the current best validation loss to do not incur into a patience penalty
        save_to: string
            Path where to save the state of the best model resulting from training.
            If early stopping is enabled, saves the model with the lowest validation loss. Otherwise, saves the model corresponding to the last iteration.
        load_from: string
            Path from where to load the state of a previously saved model.
        '''
        self.predict = None
        self.update = None
        self.error_during_train = False
        itemids = train_data[self.item_key].unique()
        self.n_items = len(itemids)
        self.init_params()  # initialize the network
        if load_from:
            logger.info('Resuming from state: {}'.format(load_from))
            self.load_state(pickle.load(open(load_from, 'rb')))

        if not retrain:
            self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
            train_data = pd.merge(train_data,
                                  pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                                  on=self.item_key, how='inner')
            user_indptr, offset_sessions = self.preprocess_data(train_data)
        else:
            raise Exception('Not supported yet!')

        if valid_data is not None:
            valid_data = pd.merge(valid_data,
                                  pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                                  on=self.item_key, how='inner')
            user_indptr_valid, offset_sessions_valid = self.preprocess_data(valid_data)

        X, Y = T.ivectors(2)
        Sstart, Ustart = T.fvectors(2)
        Hs_new, Hu_new, Y_pred, sampled_params = self.model(X, Sstart, Ustart, self.Hs, self.Hu, Y,
                                                            drop_p_hidden_usr=self.dropout_p_hidden_usr,
                                                            drop_p_hidden_ses=self.dropout_p_hidden_ses,
                                                            drop_p_init=self.dropout_p_init)
        cost = self.loss_function(Y_pred)
        # set up the parameter and sampled parameter vectors
        if self.item_embedding is None:
            params = [self.Ws_in[1:], self.Ws_hh, self.Ws_rz, self.Bs_h, self.Ws_init, self.Bs_init,
                      self.Wu_in, self.Wu_hh, self.Wu_rz, self.Bu_h]
            full_params = [self.Ws_in[0], self.Wsy, self.By]
        else:
            params = [self.Ws_in, self.Ws_hh, self.Ws_rz, self.Bs_h, self.Ws_init, self.Bs_init,
                      self.Wu_in, self.Wu_hh, self.Wu_rz, self.Bu_h]
            full_params = [self.E_item, self.Wsy, self.By]

        if self.user_propagation_mode == 'all':
            params.append(self.Wu_to_s)
        sidxs = [X, Y, Y]
        if self.user_to_output:
            full_params.append(self.Wuy)
            sidxs.append(Y)

        updates = self.RMSprop(cost, params, full_params, sampled_params, sidxs)
        eval_updates = OrderedDict()
        # Update the hidden states of the Session GRU
        for i in range(len(self.Hs)):
            updates[self.Hs[i]] = Hs_new[i]
            eval_updates[self.Hs[i]] = Hs_new[i]
        # Update the hidden states of the User GRU
        for i in range(len(self.Hu)):
            updates[self.Hu[i]] = Hu_new[i]
            eval_updates[self.Hu[i]] = Hu_new[i]

        # Compile the training and evaluation functions
        self.train_function = function(inputs=[X, Sstart, Ustart, Y], outputs=cost, updates=updates,
                                       allow_input_downcast=True,
                                       on_unused_input='warn')
        self.eval_function = function(inputs=[X, Sstart, Ustart, Y], outputs=cost, updates=eval_updates,
                                      allow_input_downcast=True,
                                      on_unused_input='warn')
        # Negative item sampling
        if self.n_sample:
            self.neg_sampler = Sampler(train_data,
                                       self.n_sample,
                                       rng=self.rng,
                                       item_key=self.item_key,
                                       sample_alpha=self.sample_alpha,
                                       sample_store=sample_store)
        # Training starts here
        best_valid, best_state = None, None
        my_patience = patience
        epoch = 0
        while epoch < self.n_epochs and my_patience > 0:
            train_cost = self.iterate(train_data, self.train_function, offset_sessions, user_indptr)
            # self.print_state()
            if np.isnan(train_cost):
                return
            if valid_data is not None:
                valid_cost = self.iterate(valid_data, self.eval_function, offset_sessions_valid, user_indptr_valid)
                if best_valid is None or valid_cost < best_valid:
                    best_valid = valid_cost
                    best_state = self.save_state()
                    my_patience = patience
                elif valid_cost >= best_valid * margin:
                    my_patience -= 1
                logger.info(
                    'Epoch {} - train cost: {:.4f} - valid cost: {:.4f} (patience: {})'.format(epoch,
                                                                                               train_cost,
                                                                                               valid_cost,
                                                                                               my_patience))
            else:
                logger.info('Epoch {} - train cost: {:.4f}'.format(epoch, train_cost))
            epoch += 1
        if my_patience == 0:
            logger.info('Early stopping condition met!')
        if best_state:
            # always load the state associated with the best validation cost
            self.load_state(best_state)
        if save_to:
            if best_state:
                state = best_state
            else:
                state = self.save_state()
            logger.info('Saving model to: {}'.format(save_to))
            pickle.dump(state, open(save_to, 'wb'), pickle.HIGHEST_PROTOCOL)

    def iterate(self, data, fun, offset_sessions, user_indptr, reset_state=True):
        if reset_state:
            # Reset session layers
            for i in range(len(self.session_layers)):
                self.Hs[i].set_value(np.zeros((self.batch_size, self.session_layers[i]), dtype=theano.config.floatX),
                                     borrow=True)
            # Reset user layers
            for i in range(len(self.user_layers)):
                self.Hu[i].set_value(np.zeros((self.batch_size, self.user_layers[i]), dtype=theano.config.floatX),
                                     borrow=True)
        # variables to manage iterations over users
        n_users = len(user_indptr)
        offset_users = offset_sessions[user_indptr]
        user_idx_arr = np.arange(n_users - 1)
        user_iters = np.arange(self.batch_size)
        user_maxiter = user_iters.max()
        user_start = offset_users[user_idx_arr[user_iters]]
        user_end = offset_users[user_idx_arr[user_iters] + 1]

        # variables to manage iterations over sessions
        session_iters = user_indptr[user_iters]
        session_start = offset_sessions[session_iters]
        session_end = offset_sessions[session_iters + 1]

        sstart = np.zeros((self.batch_size,), dtype=np.float32)
        ustart = np.zeros((self.batch_size,), dtype=np.float32)
        finished = False
        n = 0
        c = []
        while not finished:
            session_minlen = (session_end - session_start).min()
            out_idx = data.ItemIdx.values[session_start]
            for i in range(session_minlen - 1):
                in_idx = out_idx
                out_idx = data.ItemIdx.values[session_start + i + 1]
                if self.n_sample:
                    sample = self.neg_sampler.next_sample()
                    y = np.hstack([out_idx, sample])
                else:
                    y = out_idx
                cost = fun(in_idx, sstart, ustart, y)
                n += 1
                # reset sstart and ustart
                sstart = np.zeros_like(sstart, dtype=np.float32)
                ustart = np.zeros_like(ustart, dtype=np.float32)
                c.append(cost)
                if np.isnan(cost):
                    logger.error('NaN error!')
                    self.error_during_train = True
                    return
            session_start = session_start + session_minlen - 1
            session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
            sstart[session_start_mask] = 1
            for idx in session_start_mask:
                session_iters[idx] += 1
                if session_iters[idx] + 1 >= len(offset_sessions):
                    finished = True
                    break
                session_start[idx] = offset_sessions[session_iters[idx]]
                session_end[idx] = offset_sessions[session_iters[idx] + 1]

            # reset the User hidden state at user change
            user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
            ustart[user_change_mask] = 1
            for idx in user_change_mask:
                user_maxiter += 1
                if user_maxiter + 1 >= len(offset_users):
                    finished = True
                    break
                user_iters[idx] = user_maxiter
                user_start[idx] = offset_users[user_maxiter]
                user_end[idx] = offset_users[user_maxiter + 1]
                session_iters[idx] = user_indptr[user_maxiter]
                session_start[idx] = offset_sessions[session_iters[idx]]
                session_end[idx] = offset_sessions[session_iters[idx] + 1]
        avgc = np.mean(c)
        return avgc

    def predict_next_batch(self, session_ids, input_item_ids, input_user_ids,
                           predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        input_user_ids : 1D array
            Contains the user IDs of the events of the batch. Every user ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if self.error_during_train: raise Exception
        if self.predict is None or self.predict_batch != batch:
            X, Y = T.ivectors(2)
            Sstart, Ustart = T.fvectors(2)
            for i in range(len(self.session_layers)):
                self.Hs[i].set_value(np.zeros((batch, self.session_layers[i]), dtype=theano.config.floatX), borrow=True)
            for i in range(len(self.user_layers)):
                self.Hu[i].set_value(np.zeros((batch, self.user_layers[i]), dtype=theano.config.floatX), borrow=True)
            if predict_for_item_ids is not None:
                Hs_new, Hu_new, yhat, _ = self.model(X, Sstart, Ustart, self.Hs, self.Hu, Y)
            else:
                Hs_new, Hu_new, yhat, _ = self.model(X, Sstart, Ustart, self.Hs, self.Hu)
            updatesH = OrderedDict()
            for i in range(len(self.Hs)):
                updatesH[self.Hs[i]] = Hs_new[i]
            for i in range(len(self.Hu)):
                updatesH[self.Hu[i]] = Hu_new[i]

            if predict_for_item_ids is not None:
                self.predict = function(inputs=[X, Sstart, Ustart, Y], outputs=yhat, updates=updatesH,
                                        on_unused_input='warn', allow_input_downcast=True)
            else:
                self.predict = function(inputs=[X, Sstart, Ustart], outputs=yhat, updates=updatesH,
                                        on_unused_input='warn', allow_input_downcast=True)
            self.current_session = np.ones(batch) * -1
            self.current_users = np.ones(batch) * -1
            self.predict_batch = batch

        session_change = session_ids != self.current_session
        self.current_session = session_ids.copy()
        user_change = input_user_ids != self.current_users
        self.current_users = input_user_ids.copy()

        in_idxs = self.itemidmap[input_item_ids]
        if predict_for_item_ids is not None:
            iIdxs = self.itemidmap[predict_for_item_ids]
            preds = np.asarray(self.predict(in_idxs, session_change, user_change, iIdxs)).T
            return pd.DataFrame(data=preds, index=predict_for_item_ids)
        else:
            preds = np.asarray(self.predict(in_idxs, session_change, user_change)).T
            return pd.DataFrame(data=preds, index=self.itemidmap.index)

    def predict_next(self, session_id, input_item_id, input_user_ids, predict_for_item_ids=None, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''

        if not input_item_id[0] in self.itemidmap.index:
            return None

        return self.predict_next_batch(session_id, input_item_id, input_user_ids, predict_for_item_ids, 1)
        # return self.predict_next_batch(np.array([session_id]), np.array([input_item_id]), np.array([input_user_ids]), predict_for_item_ids, 1)[0]

    def clear(self):
        # Clear the session parameters
        for x in range(len(self.session_layers)):
            self.Ws_in[x].set_value([[]])
            self.Ws_hh[x].set_value([[]])
            self.Ws_rz[x].set_value([[]])
            self.Bs_h[x].set_value([])
            self.Hs[x].set_value([[]])
        self.Wsy.set_value([[]])
        self.By.set_value([[]])

        # Clear the user parameters
        for x in range(len(self.user_layers)):
            self.Wu_in[x].set_value([[]])
            self.Wu_hh[x].set_value([[]])
            self.Wu_rz[x].set_value([[]])
            self.Bu_h[x].set_value([])
            self.Hu[x].set_value([[]])
        if self.user_to_output:
            self.Wuy.set_value([[]])

        # Clear User-to-Session parameters
        self.Ws_init.clear()
        self.Bs_init.clear()

        if self.user_propagation_mode == 'all':
            self.Wu_to_s.clear()
