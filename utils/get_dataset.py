import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_reader import read_vocabulary, read_lm_data, lm_data_producer


def get_dataset(dataset, batch_size, device):
    """
    Returns data iterator for each set and vocabulary
    """
    if dataset == "wiki-02":
        data_files = [".data/wikitext-2/wiki.train.tokens.sent",
                      ".data/wikitext-2/wiki.valid.tokens.sent",
                      ".data/wikitext-2/wiki.test.tokens.sent"]
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
