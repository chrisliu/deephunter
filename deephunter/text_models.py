from __future__ import print_function

import tensorflow.keras as keras
import io
import numpy as np

class PretrainedList:
    '''Latest copies of pre-trained bert models (as of June 2, 2022)

    An updated list is maintained under the official repo:
    https://github.com/google-research/bert
    '''

    # 12-layer, 768-hidden, 12-heads, 110M parameters
    base_uncased = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_uncased = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
    # 12-layer, 768-hidden, 12-heads, 110M parameters
    base_cased = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_cased = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_uncased_whole = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_cased_whole = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip"
    # 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    multilingual_uncased = "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip"
    # 102 languages, 24-layer, 1024-hidden, 16-heads, 340M parameters
    multilingual_cased = "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
    # 12-layer, 768-hidden, 12-heads, 110M parameters
    chinese = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"

text_dataset_dir = 'text'

def downlaod_imdb():
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_subdir=text_dataset_dir)
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    return dataset_dir

def load_imdb(dataset_dir):
    def load_data(data_dir):
        texts = list()
        labels = list()

        label_index = {'pos': 1, 'neg': 0}
        for label_name, label_val in label_index.items():
            label_dir = os.path.join(data_dir, label_name)
            for fname in sorted(os.listdir(label_dir)):
                fpath = os.path.join(label_dir, fname)
                with io.open(fpath, mode='r', encoding='utf8') as ifs:
                    texts.append(ifs.read())
                    labels.append(label_val)

        return texts, labels

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    train_texts, train_labels = load_data(train_dir)
    test_texts, test_labels = load_data(test_dir)

    return train_texts, train_labels, test_texts, test_labels

def get_BERTClassifier(url):
    model_path = get_pretrained(PretrainedList.base_uncased)
    paths = get_checkpoint_paths(model_path)
    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint,
                                               training=True)

    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs =  keras.layers.Dense(units=2, activation='softmax')(dense)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    vocab_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(vocab_dict)

    return model, tokenizer

def preprocess_imdb(tokenize, texts):
    SEQ_LEN = 512
    tokenized = [tokenizer.encode(t, max_len=SEQ_LEN)[0] for t in texts]
    tokenized = np.array(tokenized)
    return [tokenized, np.zeros_like(tokenized)]

if __name__ == '__main__':
    from keras_bert import (
        get_pretrained, 
        get_checkpoint_paths, 
        load_trained_model_from_checkpoint,
        load_vocabulary,
        Tokenizer
    )
    import tensorflow as tf
    import os
    import shutil
    import time

    class Timer:
        def __init__(self):
            self.__start = None
            self.__end = None
            self.__name = ""

        def start(self, name=""):
            if name == "":
                print("Starting timer")
            else:
                print(name)

            self.__start = time.time()
            self.__name = name
            return self

        def end(self):
            self.__end = time.time()
            return self

        def print_elapsed(self):
            if self.__name == "":
                print("Took {:0.2f} seconds".format(self.__end - self.__start))
            else:
                print("{} took {:0.2f} seconds".format(self.__name, 
                                                       self.__end - self.__start))
            return self
    
    timer = Timer()

    # Download the model
    timer.start("Getting model")
    model, tokenizer = get_BERTClassifier(PretrainedList.base_cased)
    timer.end().print_elapsed()

    timer.start("Getting IMDB")
    dataset_dir = downlaod_imdb()
    train_texts, train_labels, test_texts, test_labels = load_imdb(dataset_dir)
    timer.end().print_elapsed()

    timer.start("Preprocessing")
    # TODO:TEMP
    p = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in p[:2000]]
    train_labels = [train_labels[i] for i in p[:2000]]

    p = np.random.permutation(len(test_texts))
    test_texts = [test_texts[i] for i in p[:2000]]
    test_labels = [test_labels[i] for i in p[:2000]]

    train_texts = preprocess_imdb(tokenizer, train_texts)
    test_texts = preprocess_imdb(tokenizer, test_texts)
    timer.end().print_elapsed()

    timer.start("Training")
    model.fit(train_texts, train_labels, epochs=50)
    timer.end().print_elapsed()
