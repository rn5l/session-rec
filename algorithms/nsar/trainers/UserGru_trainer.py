import sys
sys.path.append("../..")  # noqa

from time import time
from math import isnan
import numpy as np
import tensorflow as tf
import datetime


class UserGruTrainer():
    def __init__(self, sess, model, config, data_loader, logger=None):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data_loader = data_loader
        self.lossnan = False

        # Init saver
        self.saver = tf.train.Saver()

        # if config.test_path is not None:
        #     self.test_loader = DataLoader(config.test_path, config)
        #     self.evaluator = UserGruEval(sess, model, config, self.test_loader)
        #     self.best_acc = 0

        self.sess.run(tf.global_variables_initializer())

    def run_training(self):
        for epoch in range(self.config['num_epoch']):
            print(datetime.datetime.now())
            start = time()
            self.data_loader.next_epoch(shuffle=True)
            epoch_loss = self.train_epoch()
            print('++ Epoch: {} - Loss: {:.5f} - Time: {:.5f} ++'.format(
                  epoch, epoch_loss, time() - start))
            if self.lossnan:
                break

    def train_epoch(self):
        losses = []
        while self.data_loader.has_next() and not self.lossnan:
            start = time()
            loss, step = self.train_step()
            losses.append(loss)
            if isnan(loss):
                self.lossnan = True
                print("loss is nan")
            if step % self.config['display_every'] == 0:
                print('Step : {} - Loss: {:.5f} ' '- Time: {:.5f}'.format(
                    step, loss, time() - start))


        return np.mean(losses)

    def train_step(self):
        batch_data = self.data_loader.next_batch()
        feed_dict = {
            self.model.user: batch_data[:, :-1, 0],
            self.model.item: batch_data[:, :-1, 1],
            self.model.day_of_week: batch_data[:, :-1, 3],
            self.model.month_period: batch_data[:, :-1, 4],
            self.model.next_items: batch_data[:, 1:, 1],
            self.model.keep_pr: self.config['keep_pr']
        }
        _, batch_loss, step = self.sess.run(self.model.get_training_vars(),
                                            feed_dict=feed_dict)
        return batch_loss, step

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print('++ Save model to {} ++'.format(save_path))

    def load(self, path):
        self.saver.restore(self.sess, path)
        print('++ Load model from {} ++'.format(path))
