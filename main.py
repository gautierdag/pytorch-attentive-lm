import os
import argparse
import time
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from model import AttentiveRNNLanguageModel
from train import evaluate, train
from utils import generate_filename, save_attention_visualization, get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch Attentive RNN Language Modeling')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=30.0, metavar='LR',
                        help='learning rate (default: 30.0)')
    parser.add_argument('--patience', type=int, default=5, metavar='P',
                        help='patience for lr decrease (default: 5)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default 10)')
    parser.add_argument('--dataset',
                        default='ptb',
                        const='ptb',
                        nargs='?',
                        choices=['wiki-02', 'ptb'],
                        help='Select which dataset (default: %(default)s)')

    parser.add_argument('--embedding-size', type=int, default=20, metavar='N',
                        help='embedding size for embedding layer (default: 20)')
    parser.add_argument('--n-layers', type=int, default=1, metavar='N',
                        help='layer size for RNN encoder (default: 1)')
    parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                        help='hidden size for RNN encoder (default: 20)')
    parser.add_argument('--positioning-embedding', type=int, default=20, metavar='N',
                        help='hidden size for positioning generator (default: 20)')
    parser.add_argument('--input-dropout', type=float, default=0.5, metavar='D',
                        help='input dropout (default: 0.5)')
    parser.add_argument('--rnn-dropout', type=float, default=0.0, metavar='D',
                        help='rnn dropout (default: 0.0)')
    parser.add_argument('--decoder-dropout', type=float, default=0.5, metavar='D',
                        help='decoder dropout (default: 0.5)')
    parser.add_argument('--clip', type=float, default=0.25, metavar='N',
                        help='value at which to clip the norm of gradients (default: 0.25)')

    parser.add_argument('--optim',
                        default='sgd',
                        const='sgd',
                        nargs='?',
                        choices=['sgd', 'adam', 'asgd'],
                        help='Select which optimizer (default: %(default)s)')

    parser.add_argument('--salton-lr-schedule',
                        help='Enables same training schedule as Salton et al. 2017 (default: False)',
                        action='store_true')

    parser.add_argument('--early-stopping-patience', type=int, default=25, metavar='P',
                        help='early stopping patience (default: 25)')

    parser.add_argument(
        '--attention', help='Enable standard attention (default: False', action='store_true')

    parser.add_argument(
        '--no-positional-attention', help='Disable positional attention (default: False', action='store_false')

    parser.add_argument(
        '--tie-weights', help='Tie embedding and decoder weights (default: False', action='store_true')

    parser.add_argument('--file-name', action="store",
                        help='Specific filename to save under (default: uses params to generate', default=False)

    args = parser.parse_args(args)

    if not args.file_name:
        args.file_name = generate_filename(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    writer = SummaryWriter('runs/' + args.file_name)

    train_iter, valid_iter, test_iter, vocab = get_dataset(dataset=args.dataset,
                                                           batch_size=args.batch_size,
                                                           device=device)

    args.vocab_size = len(vocab)

    model = AttentiveRNNLanguageModel(args.vocab_size,
                                      embedding_size=args.embedding_size,
                                      n_layers=args.n_layers,
                                      attention=args.attention,
                                      positional_attention=args.no_positional_attention,
                                      positioning_embedding=args.positioning_embedding,
                                      hidden_size=args.hidden_size,
                                      dropout_p_decoder=args.decoder_dropout,
                                      dropout_p_encoder=args.rnn_dropout,
                                      dropout_p_input=args.input_dropout,
                                      tie_weights=args.tie_weights)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        # cannot run examples since the batch size < num GPUS
        args.parallel = True
    else:
        args.parallel = False

    model.to(device)

    # Training Set Up
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, weight_decay=12e-7)
    if args.optim == 'asgd':
        optimizer = optim.ASGD(model.parameters(),
                               lr=args.lr, weight_decay=12e-7)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.0, 0.999), eps=1e-8,
                               weight_decay=12e-7, amsgrad=True)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.patience,
        verbose=True, factor=0.5)

    early_stopping_counter = 0
    best_val_loss = False

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # Loop over epochs.
        for epoch in range(1, args.epochs + 1):

            if args.salton_lr_schedule:
                current_learning_rate = 0.5 ** max(epoch - 12, 0.0)
                optimizer.param_groups[0]['lr'] = current_learning_rate

            epoch_start_time = time.time()
            train(args, model,
                  train_iter, valid_iter,
                  criterion, optimizer,
                  epoch, writer)

            # if parallel then evaluate on single gpu
            if args.parallel:
                with open('models/temp.pt', 'wb') as f:
                    # save temporary copy
                    torch.save(model.module.to(torch.device('cpu')), f)
                    # create an instance of your network
                    single_gpu_model = torch.load(f)
                    # send to single gpu
                    single_gpu_model.to(device)
                # infer on single gpu
                val_loss = evaluate(args, single_gpu_model, valid_iter,
                                    criterion, save_attention=True, epoch=epoch,
                                    vocabulary=vocab)
                test_loss = evaluate(
                    args, single_gpu_model, test_iter, criterion)

                # use multiple GPUs again
                model = torch.load(f)
                model = nn.DataParallel(model)
                model.to(device)
            else:
                val_loss = evaluate(args, model, valid_iter,
                                    criterion, save_attention=True, epoch=epoch,
                                    vocabulary=vocab)
                test_loss = evaluate(args, model, test_iter, criterion)

            # possibly update learning rate
            scheduler.step(val_loss)

            # track learning ratei
            writer.add_scalar(
                'lr', optimizer.param_groups[0]['lr'], epoch)

            writer.add_scalar('validation_loss_at_epoch', val_loss, epoch)
            writer.add_scalar('test_loss_at_epoch', test_loss, epoch)

            writer.add_scalar('validation_perplexity_at_epoch',
                              min(math.exp(min(val_loss, 7)), 1000), epoch)

            writer.add_scalar('test_perplexity_at_epoch',
                              min(math.exp(min(test_loss, 7)), 1000), epoch)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(min(val_loss, 7))))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not os.path.exists('models'):
                    os.makedirs('models')

                with open('models/{}.pt'.format(args.file_name), 'wb') as f:
                    if args.parallel:
                        torch.save(model.module.to(torch.device('cpu')), f)
                    else:
                        torch.save(mode.to(torch.device('cpu'), f))

                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            writer.add_scalar('best_validation_perplexity_at_epoch',
                              min(math.exp(min(best_val_loss, 7)), 1000), epoch)
            if early_stopping_counter >= args.early_stopping_patience:
                print("Validation loss has not improved for {}".format(
                    early_stopping_counter))
                print("Ending Training early at epoch {}".format(epoch))
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if os.path.exists('models/{}.pt'.format(args.file_name)):
        # Load the best saved model.
        with open('models/{}.pt'.format(args.file_name), 'rb') as f:
            model = torch.load(f)
            model.to(device)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.flatten_parameters()

        # Run on test data.
        test_loss = evaluate(args, model, test_iter, criterion)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    main(sys.argv[1:])
