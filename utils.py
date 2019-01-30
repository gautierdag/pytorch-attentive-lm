import torch


def generate_filename(args):
    """
    Generates a filename from arguments passed
    Used for saving model and runs
    """
    n = args.dataset + '_n_layers_' + \
        str(args.n_layers) + '_hidden_size_' + str(args.hidden_size)
    if args.tie_weights:
        n += '_tied_weights'
    if args.attention:
        n += '_attention'
    return n


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors,
    to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def save_attention_visualization(args, data_iter, batch, model):

    vocab = data_iter.dataset.fields['text'].vocab

    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        data, targets = batch.text.t(), batch.target.t().contiguous()
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, model.vocab_size)

    print(len(data_iter))
