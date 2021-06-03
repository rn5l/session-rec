import numpy as np
from tensorflow.contrib import learn
import algorithms.nextitnet.data_loader_recsys as org

# This Data_Loader file is copied online
class Data_Loader:

    def __init__(self, options, train, test, session_key, item_key, time_key, pad_test=True):

        self.pad_test = pad_test

        if options['limit_input_length'] is None or options['limit_input_length'] is False:

            org_dl = org.Data_Loader(options, train, test, session_key, item_key, time_key, pad_test)

            self.vocab_processor = org_dl.vocab_processor

            self.item = org_dl.item
            self.item_test = org_dl.item_test
            self.item_dict = org_dl.item_dict
            self.reverse_dict = org_dl.reverse_dict
            self.input_limit = org_dl.max_session_length
            self.max_session_length = org_dl.max_session_length
            return

        elif options['limit_input_length'] is True and type(options['limit_input_length']) is bool:
            self.input_limit = int( round(train.groupby(session_key).size().mean()) ) + 1
        elif type(options['limit_input_length']) is int:
            self.input_limit = int( options['limit_input_length'] ) + 1

        max_session_length_train = train.groupby(session_key).size().max()
        max_session_length_test = test.groupby(session_key).size().max()
        if max_session_length_train > max_session_length_test:
            self.max_session_length = max_session_length_train
        else:
            self.max_session_length = max_session_length_test

        print( 'Limited the network input to {} at maximum'.format(self.input_limit) )

        rolling_iid = 2
        self.item_dict = { 0: 1, '<UNK>':0 }

        index_session = train.columns.get_loc(session_key)
        index_item = train.columns.get_loc(item_key)

        session = -1
        session_items = []

        train_data = []

        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                session = row[index_session]
                session_items = []

            if not row[index_item] in self.item_dict:
                self.item_dict[row[index_item]] = rolling_iid
                rolling_iid += 1

            session_items.append( self.item_dict[row[index_item]] )

            if len( session_items ) > self.input_limit:
                train_data.append( session_items[ -self.input_limit : ] )
            elif len( session_items ) >= 2:
                train_data.append( ( self.input_limit - len(session_items) ) * [1] + session_items )

        self.item = np.array( train_data )
        self.item_test = None
        self.reverse_dict = {v: k for k, v in self.item_dict.items()}

    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text) % sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed

    def load_translation_data(self):
        source_lines = []
        target_lines = []
        for i in range(len(self.source_lines)):
            source_lines.append(self.string_to_indices(self.source_lines[i], self.source_vocab))
            target_lines.append(self.string_to_indices(self.target_lines[i], self.target_vocab))

        buckets = self.create_buckets(source_lines, target_lines)

        return buckets, self.source_vocab, self.target_vocab

    def create_buckets(self, source_lines, target_lines):

        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab
        target_vocab = self.target_vocab

        buckets = {}
        for i in range(len(source_lines)):

            source_lines[i] = np.concatenate((source_lines[i], [source_vocab['eol']]))
            target_lines[i] = np.concatenate(([target_vocab['init']], target_lines[i], [target_vocab['eol']]))

            sl = len(source_lines[i])
            tl = len(target_lines[i])

            new_length = max(sl, tl)
            if new_length % bucket_quant > 0:
                new_length = ((new_length / bucket_quant) + 1) * bucket_quant

            s_padding = np.array([source_vocab['padding'] for ctr in range(sl, new_length)])

            # NEED EXTRA PADDING FOR TRAINING.. 
            t_padding = np.array([target_vocab['padding'] for ctr in range(tl, new_length + 1)])

            source_lines[i] = np.concatenate([source_lines[i], s_padding])
            target_lines[i] = np.concatenate([target_lines[i], t_padding])

            if new_length in buckets:
                buckets[new_length].append((source_lines[i], target_lines[i]))
            else:
                buckets[new_length] = [(source_lines[i], target_lines[i])]

            if i % 1000 == 0:
                print("Loading", i)

        return buckets
