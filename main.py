import os
import argparse
import time
import math
import sys

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import WikiText2, PennTreebank

from tensorboardX import SummaryWriter

from model import AttentiveRNNLanguageModel
from train import evaluate, train
from utils import generate_filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch Attentive RNN Language Modeling')
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
    parser.add_argument('--patience', type=int, default=15, metavar='P',
                        help='patience (default: 15)')
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

    parser.add_argument('--embedding-size', type=int, default=65, metavar='N',
                        help='embedding size for embedding layer (default: 65)')
    parser.add_argument('--n-layers', type=int, default=1, metavar='N',
                        help='layer size for RNN encoder (default: 1)')

    parser.add_argument('--hidden-size', type=int, default=65, metavar='N',
                        help='hidden size for RNN encoder (default: 65)')
    parser.add_argument('--input-dropout', type=float, default=0.5, metavar='D',
                        help='input dropout (default: 0.5)')
    parser.add_argument('--rnn-dropout', type=float, default=0.0, metavar='D',
                        help='rnn dropout (default: 0.0)')
    parser.add_argument('--decoder-dropout', type=float, default=0.5, metavar='D',
                        help='decoder dropout (default: 0.5)')

    parser.add_argument(
        '--no-attention', help='Disable attention (default: False', action='store_false')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args(args)

    run_name = generate_filename(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    writer = SummaryWriter('runs/'+run_name)

    if args.dataset == 'wiki-02':
        train_iter, valid_iter, test_iter = WikiText2.iters(
            batch_size=args.batch_size, device=device)
    if args.dataset == 'ptb':
        train_iter, valid_iter, test_iter = PennTreebank.iters(
            batch_size=args.batch_size, device=device)

    vocab_size = len(train_iter.dataset.fields['text'].vocab)

    model = AttentiveRNNLanguageModel(vocab_size,
                                      embedding_size=args.embedding_size,
                                      n_layers=args.n_layers,
                                      attention=args.no_attention,
                                      hidden_size=args.hidden_size,
                                      dropout_p_decoder=args.decoder_dropout,
                                      dropout_p_encoder=args.rnn_dropout,
                                      dropout_p_input=args.input_dropout)

    model.to(device)

    # Training Set Up
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.0, 0.999), eps=1e-8, weight_decay=12e-7)

    criterion = nn.CrossEntropyLoss()

    iteration_step = 0
    best_val_loss = 1000
    num_checkpoints = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # Loop over epochs.
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            iteration_step, val_loss, num_checkpoints = train(args, model, train_iter,
                                                              valid_iter,
                                                              criterion, optimizer,
                                                              iteration_step, epoch, best_val_loss,
                                                              num_checkpoints,
                                                              writer)

            val_loss = evaluate(model, valid_iter, criterion)
            test_loss = evaluate(model, test_iter, criterion)

            writer.add_scalar('validation_loss_at_epoch', val_loss, epoch)
            writer.add_scalar('test_loss_at_epoch', test_loss, epoch)
            writer.add_scalar('validation_perplexity_at_epoch',
                              math.exp(val_loss), epoch)
            writer.add_scalar('test_perplexity_at_epoch',
                              math.exp(test_loss), epoch)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if args.save_model:
                    if not os.path.exists('models'):
                        os.makedirs('models')

                    with open('models/{}.pt'.format(run_name), 'wb') as f:
                        torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if args.save_model and os.path.exists('models/{}.pt'.format(run_name)):
        # Load the best saved model.
        with open('models/{}.pt'.format(run_name), 'rb') as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.flatten_parameters()

        # Run on test data.
        test_loss = evaluate(model, test_iter, criterion)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main(sys.argv[1:])
