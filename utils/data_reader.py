# -*- coding: utf-8 -*-
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
import codecs
import collections
import math
import os
import re

import numpy as np


# Special vocabulary symbols - we always put them at the start.
_PAD = '<PAD>'
_EOS = '<eos>'
_UNK = '<unk>'

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

START_VOCAB = [_PAD, _EOS, _UNK]

_DIGIT_RE = re.compile(r'\d')


def _read_words(filename, ptb=True):
    """ helper to read the file and split into sentences. """

    sentences = []
    with codecs.open(filename, "rb", "utf-8") as f:
        sentences = f.readlines()

    sentences = [sent for sent in sentences if len(sent) > 0]

    if ptb:
        return [sentence.strip().split() + [_EOS]
                for sentence in sentences]
    else:
        return [sentence.strip().split()
                for sentence in sentences]


def read_vocabulary(data_filenames, vocab_size):
                    # normalize_digits=True):
    """ Helper to build the vocabulary from the given filename. It makes use
    of python collections.Counter to help counting the occurrences of each word.
    """
    lines = []
    for filename in data_filenames:
        for line in codecs.open(filename, "r", "utf-8"):
            lines.append(line)

    words = []
    for line in lines:
        words += line.split()

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    ws, _ = list(zip(*count_pairs))
    ws = [w for w in ws if w not in START_VOCAB]
    words = START_VOCAB + [w for w in ws if w not in START_VOCAB]
    words = words[:vocab_size]

    word_dict = dict(zip(words, range(len(words))))

    return word_dict


def sentence_to_token_ids(sentence,
                          vocabulary):
    """ Convert a string to list of integers representing token-ids. """
    return [vocabulary.get(w, UNK_ID) for w in sentence]


def _data_to_token_ids(data_path,
                       vocab,
                       print_info=True):
    """ Tokenize data file and turn into token-ids using given the vocabulary. """
    dataset = _read_words(data_path)
    tokens = [sentence_to_token_ids(sentence, vocab) for sentence in dataset]
    n_words = sum([len(tok) for tok in tokens])
    if print_info:
        n_samples = len(tokens)
        print("  # of sentences : {0}".format(n_samples))
    return tokens, n_words


def read_lm_data(data_files,
                 vocabulary,
                 print_info=True):
    """Read datasets from data_path, build the vocabulary based on the training
    dataset and convert words in traning, validation and test sets into
    python integers.
    """

    train_path = data_files[0]
    valid_path = data_files[1]
    test_path = data_files[2]

    print("\nReading training data from {0}".format(train_path))
    train_data, _ = _data_to_token_ids(
        train_path, vocabulary, print_info=print_info)

    print("\nReading validation data from {0}".format(valid_path))
    valid_data, _ = _data_to_token_ids(
        valid_path, vocabulary, print_info=print_info)

    print("\nReading test data from {0}".format(test_path))
    test_data, _ = _data_to_token_ids(
        test_path, vocabulary, print_info=print_info)

    return train_data, valid_data, test_data


def lm_data_producer(raw_data,
                     num_steps=35,
                     dtype=np.long):
    """ Iterate on the given raw data producing samples. """

    # we pad or cut the sentences to be of length num_steps
    raw_data = [sentence + [PAD_ID] * (num_steps + 1 - len(sentence))
                if len(sentence) < num_steps + 1 else sentence[0:(num_steps + 1)]
                for sentence in raw_data]

    data = np.array(raw_data, dtype=dtype)

    # raw_data = np.array(raw_data, dtype=dtype)
    # data_len = len(raw_data)
    # epoch_size = int(math.ceil(data_len / batch_size))
    # data = np.zeros_like(raw_data, dtype=dtype)

    # for i in range(epoch_size):
    #     dt = raw_data[batch_size * i:batch_size * (i + 1)]
    #     data[batch_size * i:batch_size * (i + 1)] = dt
    #     # Append empty to make data fit into even batches
    #     if len(dt) < batch_size:
    #         shape = (batch_size - len(dt)), num_steps + 1
    #         padding = np.ones(shape) * PAD_ID
    #         data = np.concatenate([data, padding])

    xtrain = data[:, 0:num_steps].astype(dtype)
    ytrain = data[:, 1:num_steps + 1].astype(dtype)

    return xtrain, ytrain
