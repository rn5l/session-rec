import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import datetime
import time
from .model_last import Graph, parse_function_, run_epoch, eval_epoch,random_name, random_validation

class Trainer():

    def __init__(self, opt, n_item, n_user):

        self.n_item = n_item
        self.n_user = n_user
        self.data_path = './temp/tfrecord'
        self.opt = opt

        self.padded_shape = {'A_in': [None, None],
                        'A_out': [None, None],
                        'session_alias_shape': [None],
                        'session_alias': [None, None],
                        'seq_mask': [],
                        'session_len': [],
                        'tar': [],
                        'user': [],
                        'session_mask': [None],
                        'seq_alias': [None],
                        'num_node': [],
                        'all_node': [None],
                        'A_in_shape': [None],
                        'A_out_shape': [None],
                        'A_in_row': [None],
                        'A_in_col': [None],
                        'A_out_row': [None],
                        'A_out_col': [None]}
        # --------------------------从文件中读取文件名-------------------------------

        self.train_filenames = random_name(self.data_path + '/' + 'user_*/' + 'train_*.tfrecord')
        self.train_dataset = tf.data.TFRecordDataset(self.train_filenames)
        self.train_dataset = self.train_dataset.map(parse_function_(self.opt.max_session)).shuffle(buffer_size=self.opt.buffer_size)

        self.train_batch_padding_dataset = self.train_dataset.padded_batch(self.opt.batchSize, padded_shapes=self.padded_shape,
                                                                 drop_remainder=True)
        self.train_iterator = self.train_batch_padding_dataset.make_initializable_iterator()

        self.model = Graph(hidden_size=opt.hiddenSize, user_size=opt.userSize, batch_size=opt.batchSize,
                      seq_max=opt.max_length,
                      group_max=opt.max_session,
                      n_item=n_item, n_user=n_user, lr=opt.lr,
                      l2=opt.l2, step=opt.step, decay=opt.decay, ggnn_drop=opt.ggnn_drop, graph=opt.graph,
                      mode=opt.mode,
                      data=opt.data, decoder_attention=opt.decoder_attention, encoder_attention=opt.encoder_attention,
                      behaviour_=opt.behaviour_, pool=opt.pool)

        self.train_data = self.train_iterator.get_next()

        with tf.variable_scope('model', reuse=None):
            self.train_loss, self.train_opt = self.model.forward(self.train_data['A_in'], self.train_data['A_out'], self.train_data['all_node'],
                                                  self.train_data['seq_alias'], self.train_data['seq_mask'],
                                                  self.train_data['session_alias'],
                                                  self.train_data['session_len'], self.train_data['session_mask'],
                                                  self.train_data['tar'],
                                                  self.train_data['user'])
        print("Prima del training")

    def train(self, sess):

        step = 0
        date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        if self.opt.user_:
            csvfile = str(date_now) + self.opt.graph + '_' + self.opt.data + '_' +str(self.opt.max_session)+'_'+str(self.opt.max_length)+'_d' +\
                      str(self.opt.hiddenSize) + '_u' + str(self.opt.userSize) + '_' + self.opt.mode

            his_csvfile = str(date_now) + self.opt.graph + '_history_' + self.opt.data + '_' + str(self.opt.max_session) + '_' + str(
                          self.opt.max_length) + '_d' + str(self.opt.hiddenSize) + '_u' + str(self.opt.userSize) + '_' + self.opt.mode
        else:
            csvfile = str(date_now) + self.opt.graph + '_' + self.opt.data + '_' + str(self.opt.max_session) + '_' + str(self.opt.max_length) + '_d' + str(
                      self.opt.hiddenSize) + '_' + self.opt.mode
            his_csvfile = str(date_now) + self.opt.graph + '_history_'  + self.opt.data + '_' + str(self.opt.max_session) + '_' + str(
                          self.opt.max_length) + '_d' + str(self.opt.hiddenSize) + '_' + self.opt.mode
        for epoch in range(self.opt.epoch):
            sess.run([self.train_iterator.initializer])
            print('epoch: ', epoch, '====================================================')
            print('start training: ', datetime.datetime.now())
            step, mean_train_loss = run_epoch(sess, self.train_loss, self.train_opt, step, max_length=self.opt.max_length, max_session=self.opt.max_session)
            print('step: %d\ttrain_loss: %.4f\tEpoch: %d' %
                  ( step,mean_train_loss, epoch))

        print("dopo il training")
