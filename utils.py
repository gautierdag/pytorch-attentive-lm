import numpy as np
import torch
from sys import platform as sys_pf

if sys_pf == 'darwin':
    print("Use darwin detected")
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt


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

    data, targets = batch.text.t(), batch.target.t().contiguous()
    output, hidden, attention_weights = model(
        data, hidden, return_attention=True)

    output_flat = output.view(-1, model.vocab_size)

    plot_attention(vocab, data, targets, attention_weights)


def plot_attention(vocab, data, targets, attention_weights):
    batch_size = data.shape[0]
    seq_length = data.shape[1]

    EXAMPLE_INDEX = 10

    clean_attention_weights = np.zeros((seq_length, seq_length))
    for a in range(len(attention_weights)):
        np_a = attention_weights[a][EXAMPLE_INDEX].cpu(
        ).detach().numpy().flatten()
        for w in range(len(np_a)):
            clean_attention_weights[a][w] = np_a[w]

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(clean_attention_weights, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + [vocab.itos[data[EXAMPLE_INDEX][i]]
                               for i in range(seq_length)], rotation=90)
    ax.set_yticklabels([''] + [vocab.itos[targets[EXAMPLE_INDEX][i]]
                               for i in range(seq_length)])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)


# evaluateAndShowAttention("elle a cinq ans de moins que moi .")

# evaluateAndShowAttention("elle est trop petit .")

# evaluateAndShowAttention("je ne crains pas de mourir .")

# evaluateAndShowAttention("c est un jeune directeur plein de talent .")
