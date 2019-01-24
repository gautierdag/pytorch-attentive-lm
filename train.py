import math
import torch
import time
from tqdm import tqdm


def evaluate(model, data_iterator, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    example_count = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_iterator), total=len(data_iterator), disable=True):
            data, targets = batch.text.t(), batch.target.t().contiguous()
            output = model(data)
            output_flat = output.view(-1, model.vocab_size)
            total_loss += len(data) * criterion(output_flat,
                                                targets.view(-1)).item()
            example_count += len(data)

    model.train()
    return total_loss / example_count


def train(args, model, train_iter, valid_iter,
          criterion, optimizer,
          iteration_step, epoch,
          writer,):

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_num_examples = 0
    start_time = time.time()

    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), disable=True):
            # transpose text to make batch first
        iteration_step += 1
        data, targets = batch.text.t(), batch.target.t().contiguous()

        model.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # Note this shouldn't be truly necessery here since we do not propagate the hidden states over mini batches
        # However they have be shown to behave better in steep cliffs loss surfaces
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        total_loss += len(data)*loss.item()
        total_num_examples += len(data)

        if iteration_step % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / total_num_examples
            elapsed = time.time() - start_time
            exp_cur_loss = min(cur_loss, 7)
            # note printing innacurat perplexity if high - to not clog ouput
            print('| epoch {:3d} | {}/{} batches | ms/batch {:5.2f} \
                | loss {:5.2f} | ppl {:8.2f}'.format(epoch,
                                                     iteration_step, len(
                                                         train_iter)*args.epochs,
                                                     elapsed * 1000 / args.log_interval,
                                                     cur_loss, min(math.exp(exp_cur_loss), 1000)))
            writer.add_scalar('training_loss', cur_loss, iteration_step)
            writer.add_scalar('training_perplexity',
                              min(math.exp(exp_cur_loss), 1000), iteration_step)
            total_loss = 0
            total_num_examples = 0
            start_time = time.time()

    return iteration_step
