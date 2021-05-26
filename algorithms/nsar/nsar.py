import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np

from algorithms.nsar.data_loader.data_loader import DataLoader
from algorithms.nsar.models.UserGru import UserGruModel

from algorithms.nsar.trainers.UserGru_trainer import UserGruTrainer

SAVE_MODEL_FOLDER = "algorithms/filemodel/saved/nsar/"

class NSAR:
    def __init__(self, cell='gru', name='baseline', context_embedding=5, hidden_units=100,
                 num_layers=1, combination='adaptive',
                 fusion_type='post', learning_rate=0.001, num_epoch=20,
                 display_every=500, eval_every=1, keep_pr=0.25, batch_size=64, mode='train',
                 session_key='SessionId', item_key='ItemId', user_key ='UserId', time_key='Time'):


        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key

        self.mode = mode
        self.name = name
        self.combination = combination
        self.fusion_type = fusion_type

        # Hyper params
        self.cell = cell
        self.num_layers = num_layers
        self.entity_embedding = hidden_units
        self.context_embedding = context_embedding
        self.hidden_units = hidden_units
        
        # Learning params
        self.learning_rate = learning_rate
        self.keep_pr = keep_pr
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        # Logging
        self.display_every = display_every
        self.eval_every = eval_every



    def fit(self, train_data, test_data=None):

        self.sess = self.get_tensorflow_session()
        max_train_length = train_data.groupby([self.session_key])[self.item_key].count().max()
        max_test_length = test_data.groupby([self.session_key])[self.item_key].count().max()
        self.max_length = max(max_train_length, max_test_length)

        self.itemids = train_data[self.item_key].unique()
        print("train data items: "+str(len(self.itemids)))


        assert(len(np.setdiff1d(test_data[self.item_key].unique(),self.itemids, assume_unique=True)) == 0)
        self.userids = train_data[self.user_key].unique()
        assert(len(np.setdiff1d(test_data[self.user_key].unique(), self.userids, assume_unique=True)) == 0)
#        assert((np.union1d(train_data[self.user_key].unique(), test_data[self.user_key].unique()) == self.userids).all())

        self._num_items = len(self.itemids)
        self.item2id = dict(zip(self.itemids, range(1, len(self.itemids) + 1)))
        self.user2id = dict(zip(self.userids, range(1, len(self.userids) + 1)))

        self._num_users = len(self.user2id)

        self.data = train_data

        print("++PARAMETERS++ \nnum_epoch: "+ str(self.num_epoch) + "\n" \
              "learning_rate: "+ str(self.learning_rate)+"\n"
              "hidden_units: "+ str(self.hidden_units)+"\n")
        config = dict()
        config['num_users'] = self._num_users
        config['num_items'] = self._num_items
        config['max_length'] = self.max_length
        config['cell'] = self.cell
        config['entity_embedding'] = self.entity_embedding
        config['context_embedding'] = self.context_embedding
        config['hidden_units'] = self.hidden_units
        config['num_layers'] = self.num_layers
        config['combination'] = self.combination
        config['fusion_type'] = self.fusion_type
        config['learning_rate'] = self.learning_rate
        config['name'] = self.name
        config['num_epoch'] = self.num_epoch
        config['keep_pr'] = self.keep_pr
        config['batch_size'] = self.batch_size
        config['display_every'] = self.display_every
        config['eval_every'] = self.eval_every

        self.model = UserGruModel(config)

        self.train_loader = DataLoader(config)

        self.train_loader.load_data(train_data, self.user2id, self.item2id)

        self.trainer = UserGruTrainer(self.sess, self.model, config, self.train_loader)


        self.trainer.run_training()

        self.test_loader = DataLoader(config)
        self.test_loader.load_data(test_data, self.user2id, self.item2id)
        self.current_session = -1


    def get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def get_tensorflow_session(self):
        config = tf.ConfigProto(device_count={'GPU': 1})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def predict_next(self, session_id, input_item_id, input_user_id, predict_for_item_ids=None, timestamp=None):

        if self.current_session != session_id:
            self.current_session = session_id
            self.pos = 0

            session = self.test_loader.data_from_sid(session_id)

            feed_dict = {
                self.model.user: session[:, :-1, 0],
                self.model.item: session[:, :-1, 1],
                self.model.day_of_week: session[:, :-1, 3],
                self.model.month_period: session[:, :-1, 4],
                self.model.next_items: session[:, 1:, 1],
                self.model.keep_pr: 1
            }

            self.pr = self.sess.run(self.model.get_output(), feed_dict=feed_dict)
            assert len(self.pr) != 1
        else:
            self.pos += 1

        pr = np.asanyarray(self.pr[self.pos,1:])
        prob = pd.DataFrame(data=pr, index=list(self.item2id.keys()))[0]
        #if predict_for_item_ids is not None:
        #    prob = prob.loc[prob.index.isin(predict_for_item_ids)]
        return prob

    def support_users(self):
        '''
          whether it is a session-based or session-aware algorithm
          (if returns True, method "predict_with_training_data" must be defined as well)
          Parameters
          --------
          Returns
          --------
          True : if it is session-aware
          False : if it is session-based
        '''
        return True


    def predict_with_training_data(self):
        '''
            (this method must be defined if "support_users is True")
            whether it also needs to make prediction for training data or not (should we concatenate training and test data for making predictions)

            Parameters
            --------

            Returns
            --------
            True : e.g. hgru4rec
            False : e.g. uvsknn
            '''
        return False

    def clear(self):
        self.current_session = -1
        self.sess.close()
        tf.reset_default_graph()
        pass