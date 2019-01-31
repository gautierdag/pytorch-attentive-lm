import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import urllib.request

from .data_reader import read_vocabulary, read_lm_data, lm_data_producer
from .pre_process_wikitext import pre_process


def get_dataset(dataset, batch_size, device):
    """
    Returns data iterator for each set and vocabulary
    """
    download_dataset(dataset)  # downloads and preprocess dataset if needed
    if dataset == "wiki-02":
        data_files = [".data/wikitext-2/wikitext-2/wiki.train.tokens.sent",
                      ".data/wikitext-2/wikitext-2/wiki.valid.tokens.sent",
                      ".data/wikitext-2/wikitext-2/wiki.test.tokens.sent"]
        vocab_size = 33278 + 1  # add 1 to account for PAD
    if dataset == 'ptb':
        data_files = [".data/penn-treebank/ptb.train.txt",
                      ".data/penn-treebank/ptb.valid.txt",
                      ".data/penn-treebank/ptb.test.txt"]
        vocab_size = 10000 + 1  # add 1 to account for PAD

    vocabulary = read_vocabulary(data_files, vocab_size)

    train_data, valid_data, test_data = read_lm_data(data_files,
                                                     vocabulary)

    # Convert numpy to datasets and obtain iterators for each
    train_data = lm_data_producer(train_data)
    train_x = torch.tensor(train_data[0], dtype=torch.long, device=device)
    train_y = torch.tensor(train_data[1], dtype=torch.long, device=device)
    train_dataset = TensorDataset(train_x, train_y)

    valid_data = lm_data_producer(valid_data)
    valid_x = torch.tensor(valid_data[0], dtype=torch.long, device=device)
    valid_y = torch.tensor(valid_data[1], dtype=torch.long, device=device)
    valid_dataset = TensorDataset(valid_x, valid_y)

    test_data = lm_data_producer(test_data)
    test_x = torch.tensor(test_data[0], dtype=torch.long, device=device)
    test_y = torch.tensor(test_data[1], dtype=torch.long, device=device)
    test_dataset = TensorDataset(test_x, test_y)

    train_iter = DataLoader(train_dataset, batch_size=batch_size)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size)
    test_iter = DataLoader(test_dataset, batch_size=batch_size)

    return train_iter, valid_iter, test_iter, vocabulary


# downloading/preprocessing functions
def download_dataset(dataset):
    if not os.path.exists('.data'):
        os.makedirs('.data')
    if dataset == 'ptb':
        folder_name = 'penn-treebank'
        filename = 'ptb.test.txt'
    if dataset == 'wiki-02':
        folder_name = 'wikitext-02'
        filename = 'wiki.test.tokens'
    dataset_path = '.data/' + folder_name
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    filepath = dataset_path + '/' + filename
    if not os.path.exists(filepath):
        if dataset == 'ptb':
            download_ptb(dataset_path)
        if dataset == 'wikitext-2':
            download_and_preproc_wiki(dataset_path)
    return


def download_ptb(dataset_path):
    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']

    urllib.request.retrieve(urls[0], dataset_path+"/ptb.train.txt")
    urllib.request.retrieve(urls[1], dataset_path+"/ptb.valid.txt")
    urllib.request.retrieve(urls[2], dataset_path+"/ptb.test.txt")


def download_and_preproc_wiki(dataset_path):
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    raise NotImplementedError

    train = ".data/wikitext-2/wikitext-2/wiki.train.tokens"
    valid = ".data/wikitext-2/wikitext-2/wiki.valid.tokens"
    test = ".data/wikitext-2/wikitext-2/wiki.test.tokens"

    print("Pre-processing wikitext-02 training set...")
    pre_process(train)

    print("Pre-processing wikitext-02 validation set...")
    pre_process(valid)

    print("Pre-processing wikitext-02 test set...")
    pre_process(test)
