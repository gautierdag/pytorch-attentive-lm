import os
import argparse
import time
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import WikiText2, PennTreebank

from model import AttentiveRNNLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_iterator):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    example_count = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_iterator), total=len(data_iterator), disable=True):
            data, targets = batch.text.t(), batch.target.t().contiguous()
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat,
                                                targets.view(-1)).item()
            example_count += len(data)

    return total_loss / example_count


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--optimization-step', type=int, default=100, metavar='N',
                        help='number of steps between optimizing learning rate')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset',
                        default='ptb',
                        const='ptb',
                        nargs='?',
                        choices=['wiki-02', 'ptb'],
                        help='Select which dataset (default: %(default)s)')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'wiki-02':
        train_iter, valid_iter, test_iter = WikiText2.iters(
            batch_size=args.batch_size, device=device)
    if args.dataset == 'ptb':
        train_iter, valid_iter, test_iter = PennTreebank.iters(
            batch_size=args.batch_size, device=device)

    vocab_size = len(train_iter.dataset.fields['text'].vocab)

    model = AttentiveRNNLanguageModel(vocab_size)
    model.to(device)

    # Training Set Up
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.0, 0.999), eps=1e-8, weight_decay=12e-7)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    best_val_loss = None

    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        total_num_examples = 0
        start_time = time.time()

        number_of_checkpoints_since_last_loss_decrease = 0
        best_val_loss = 1000

        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), disable=True):
                # transpose text to make batch first
            data, targets = batch.text.t(), batch.target.t().contiguous()

            model.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += len(data)*loss.item()
            total_num_examples += len(data)

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / total_num_examples
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {}/{} batches | ms/batch {:5.2f} \
                    | loss {:5.2f} | ppl {:8.2f}'.format(epoch,
                                                         i, len(train_iter),
                                                         elapsed * 1000 / args.log_interval,
                                                         cur_loss, math.exp(cur_loss)))
                total_loss = 0
                total_num_examples = 0
                start_time = time.time()

            if i % args.optimization_step == 0 and i > 0:
                loss = evaluate(valid_iter)
                if loss < best_val_loss:
                    best_val_loss = loss
                    number_of_checkpoints_since_last_loss_decrease = 0
                else:
                    number_of_checkpoints_since_last_loss_decrease += 1

                if number_of_checkpoints_since_last_loss_decrease >= 30:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    print("30 checkpoints since last decrease - decreasing lr rate")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_decay

                    number_of_checkpoints_since_last_loss_decrease = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(valid_iter)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if args.save_model:
                    with open('best_model.pt', 'wb') as f:
                        torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if args.save_model:
        # Load the best saved model.
        with open('best_model.pt', 'rb') as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.flatten_parameters()

        # Run on test data.
        test_loss = evaluate(test_iter)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main()
