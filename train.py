import math
import torch
import time


from utils import repackage_hidden, save_attention_visualization


def evaluate(args, model, data_iterator, criterion,
             save_attention=False, epoch=0):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    example_count = 0

    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for _, batch in enumerate(data_iterator):
            data, targets = batch[0], batch[1]
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, model.vocab_size)
            total_loss += len(data) * criterion(output_flat,
                                                targets.view(-1)).item()
            example_count += len(data)
            hidden = repackage_hidden(hidden)
    if save_attention and (args.attention or args.no_positional_attention):
        save_attention_visualization(args, data_iterator, batch, model, epoch)
    model.train()
    return total_loss / example_count


def train(args, model, train_iter, valid_iter,
          criterion, optimizer,
          epoch, writer):

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_num_examples = 0
    start_time = time.time()
    iteration_step = len(train_iter) * (epoch - 1)

    hidden = model.init_hidden(args.batch_size)
    for i, batch in enumerate(train_iter):
            # transpose text to make batch first
        iteration_step += 1
        data, targets = batch[0], batch[1]

        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # Note this shouldn't be truly necessery here since we do not propagate the hidden states over mini batches
        # However they have be shown to behave better in steep cliffs loss surfaces
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # We detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start
        # of the dataset.
        hidden = repackage_hidden(hidden)

        total_loss += len(data) * loss.item()
        total_num_examples += len(data)

        if iteration_step % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / total_num_examples
            elapsed = time.time() - start_time
            exp_cur_loss = min(cur_loss, 7)
            # note printing innacurat perplexity if high - to not clog ouput
            print('| epoch {:3d} | {}/{} batches | ms/batch {:5.2f} \
                | loss {:5.2f} | ppl {:8.2f}'.format(epoch,
                                                     iteration_step, len(
                                                         train_iter) * args.epochs,
                                                     elapsed * 1000 / args.log_interval,
                                                     cur_loss, min(math.exp(exp_cur_loss), 1000)))
            writer.add_scalar('training_loss', cur_loss, iteration_step)
            writer.add_scalar('training_perplexity',
                              min(math.exp(exp_cur_loss), 1000), iteration_step)
            total_loss = 0
            total_num_examples = 0
            start_time = time.time()
