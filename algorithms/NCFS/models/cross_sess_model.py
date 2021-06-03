from keras.layers import Input, add, dot, multiply, concatenate, GRU, Lambda, Embedding, Dense, TimeDistributed, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K, regularizers
from .attentionlayer import Attention, PairAttention


class CrossSessRS(object):
    def __init__(self,
                 num_items,
                 neg_samples,
                 embedding_len=200,
                 ctx_len = 10,
                 max_sess_len=20,
                 max_nb_sess=10,
                 dropout=0,
                 att_alpha=1):
        self.num_items = num_items
        self.neg_samples = neg_samples
        self.embedding_len = embedding_len
        self.ctx_len = ctx_len
        self.max_sess_len = max_sess_len
        self.max_nb_sess = max_nb_sess
        self.use_his_session = max_nb_sess > 0
        self.dropout = dropout
        self.att_alpha = att_alpha
        self.sess_embedding = Embedding(self.num_items+1, self.embedding_len, mask_zero=True, name='item_embedding')
        self._build()

    def _build(self):
        sess_input = Input(shape=(self.max_sess_len,), dtype='int32', name='sess_index')
        embedded_sequences = self.sess_embedding(sess_input)

        his_session_embed = Attention(name='his_sess_attention', alpha=self.att_alpha)(embedded_sequences)
        sess_encoder = Model(sess_input, his_session_embed, name='model_sess_encoder')

        # historical session encoding
        history_input = Input(shape=(self.max_nb_sess, self.max_sess_len), dtype='int32', name='his_sess_index')
        sessions = TimeDistributed(sess_encoder)(history_input)
        his_sess_embed = GRU(self.embedding_len, activation='relu',
                             dropout=self.dropout, recurrent_dropout=self.dropout, name='his_sess_embed')(sessions)

        # current attention
        curr_sess_input = Input(shape=(self.ctx_len,), dtype='int32', name='curr_sess_index')
        curr_sess_embed = self.sess_embedding(curr_sess_input)
        # his_sess_embed = Lambda(lambda x : K.expand_dims(x, 1))(his_sess_embed)
        ctx_embed = Attention(name='cross_sess_attention', alpha=self.att_alpha)(curr_sess_embed)

        if self.use_his_session:
            his_sess_embed = BatchNormalization()(his_sess_embed)
            ctx_embed = BatchNormalization()(ctx_embed)
            concat_rep = concatenate([ctx_embed, his_sess_embed])
            hist_gate = Dense(self.embedding_len, activation='sigmoid')(concat_rep)
            curr_gate = Lambda(lambda x: 1 - x) (hist_gate)
            his_gate_embed = multiply([hist_gate, his_sess_embed])
            curr_gate_embed = multiply([curr_gate, ctx_embed])
            ctx_embed = add([his_gate_embed, curr_gate_embed])

        target_input = Input(shape=(self.neg_samples + 1,), dtype='int32', name='target_index')
        pred_embedding = Embedding(self.num_items + 1, K.int_shape(ctx_embed)[-1], name='target_embedding')
        target_embed = pred_embedding(target_input)

        prob = Activation('softmax')(dot([target_embed, ctx_embed], axes=-1))
        self.train_model = Model(inputs=[curr_sess_input, history_input, target_input], outputs=prob, name='att_sess_model')

        predict_input = Input(shape=(self.num_items + 1,), dtype='int32', name='predict_all_index')
        predict_embed = pred_embedding(predict_input)
        pred_score = dot([predict_embed, ctx_embed], axes=-1)
        self.predict_model = Model(inputs=[curr_sess_input, history_input, predict_input], outputs=pred_score, name='pred_sess_model')



