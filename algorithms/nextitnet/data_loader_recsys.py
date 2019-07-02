import numpy as np
from tensorflow.contrib import learn


# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options, train, test,session_key,item_key,time_key, pad_test=True):

        '''positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]'''
        
        
        self.pad_test = pad_test
        
        # positive_examples=[",".join("{}:{}".format(*t) for t in zip(cols, row)) for _, row in df[["B", "C", "D"]].iterrows()]

        max_session_length_train = train.groupby(session_key).size().max()
        max_session_length_test = test.groupby(session_key).size().max()
        if max_session_length_train > max_session_length_test:
            max_session_length = max_session_length_train
        else:
            max_session_length = max_session_length_test
        print("max session lenght:" + str(max_session_length))

        self.max_session_length = max_session_length


        train_padded = (train.groupby(session_key)[item_key]
                        .apply(lambda x: list(x))
                        .reset_index())

        for index, row in train_padded.iterrows():
            row['ItemId'][:0] = [0] * (max_session_length - len(row[item_key]))
        # index_time = train.columns.get_loc(self.time_key)
        '''
        Session_id   padded_list_items   Joined
        1  [0, 5, 6, 6]  0,5,6,6
        2  [0, 0, 0, 8]  0,0,0,8
        '''
        train_padded['Joined'] = train_padded[item_key].apply(lambda x: ','.join(map(str, x)))

        # ['0,5,6,6', '0,0,0,8']
        positive_examples = train_padded['Joined'].tolist()
        max_document_length = max([len(x.split(",")) for x in positive_examples])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

        self.item = np.array(list(self.vocab_processor.fit_transform(positive_examples)))
        self.item_test = self.transform_test(test,session_key,item_key,time_key)
        self.item_dict = self.vocab_processor.vocabulary_._mapping
        self.reverse_dict = {v: k for k, v in self.item_dict.items()}

    def transform_test(self, test,session_key,item_key,time_key):
        #max_session_length = test.groupby(session_key).size().max()

        train_padded = (test.groupby(session_key)[item_key]
                        .apply(lambda x: list(x))
                        .reset_index())

        
        if self.pad_test:
            for index, row in train_padded.iterrows():
               row['ItemId'][:0] = [0] * (self.max_session_length - len(row[item_key]))


        # index_time = train.columns.get_loc(self.time_key)
        '''
        Session_id   ItemId   Joined
        1  [0, 5, 6, 6]  0,5,6,6
        2  [0, 0, 0, 8]  0,0,0,8
        '''
        train_padded['Joined'] = train_padded[item_key].apply(lambda x: ','.join(map(str, x)))

        # ['0,5,6,6', '0,0,0,8']
        positive_examples = train_padded['Joined'].tolist()
        return np.array(list(self.vocab_processor.fit_transform(positive_examples)))

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

    def build_vocab(self, sentences):
        vocab = {}
        ctr = 0
        for st in sentences:
            for ch in st:
                if ch not in vocab:
                    vocab[ch] = ctr
                    ctr += 1

        # SOME SPECIAL CHARACTERS
        vocab['eol'] = ctr
        vocab['padding'] = ctr + 1
        vocab['init'] = ctr + 2

        return vocab

    def string_to_indices(self, sentence, vocab):
        indices = [vocab[s] for s in sentence.split(',')]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = {vocab[ch]: ch for ch in vocab}
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)

    def get_batch_from_pairs(self, pair_list):
        source_sentences = []
        target_sentences = []
        for s, t in pair_list:
            source_sentences.append(s)
            target_sentences.append(t)

        return np.array(source_sentences, dtype='int32'), np.array(target_sentences, dtype='int32')

