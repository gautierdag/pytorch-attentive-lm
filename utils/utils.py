import numpy as np
import torch

from sys import platform as sys_pf
if sys_pf == 'darwin':
    print("Use darwin detected")
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
else:
    import matplotlib
    import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    if args.no_positional_attention:
        n += '_positional_attention'
    return n


def convert_tensor_to_sentence(vocab, tensor):
    """
    converts a 1d tensor back to a sentence
    - should be ints representing index in vocab dict
    """
    sentence = ''
    for t in tensor:
        sentence += vocab.itos[t]
        sentence += ' '
    print(sentence)


def convert_sentence_to_tensors(vocab, sentence):
    """
    converts a give sentence such as 
    sentence = "the man bought the horse which i saw in japan."
    to tensor using vocab
    """
    sentence = sentence.lower().replace('.', ' <eos>').split(' ')
    encoded = []
    for w in sentence:
        i = vocab.stoi.get(w)
        if i is None:
            i = vocab.stoi.get('<unk>')
        encoded.append(i)

    encoded = torch.tensor(encoded, dtype=torch.long, device=device)

    length = torch.tensor(len(encoded), dtype=torch.float, device=device)
    model_input = encoded[:-1]
    target = encoded[1:]

    # convert_tensor_to_sentence(vocab, model_input)
    # convert_tensor_to_sentence(vocab, target)

    return model_input.unsqueeze(0), target.unsqueeze(0), length


def save_attention_visualization(args, model, vocabulary, epoch):

    # Test on standard sentences
    sentence1 = "I saw the woman who I think he likes in japan last year."
    sentence2 = "In japan last year I saw the woman who I think he likes."
    sentence3 = "the man bought the horse which i saw."
    sentences = [sentence1, sentence2, sentence3]
    for s in range(len(sentences)):
        input_sentence, target_sentence, l = convert_sentence_to_tensors(vocabulary,
                                                                      sentences[s])
        _, attention_weights = model(
            input_sentence, l, return_attention=True)
        plot_attention(args, vocabulary, input_sentence,
                       target_sentence, attention_weights, epoch, count=s)


def plot_attention(args, vocab, data, targets,
                   attention_weights, epoch, count=0):
    batch_size = data.shape[0]
    seq_length = data.shape[1]

    clean_attention_weights = np.zeros((seq_length, seq_length))
    for a in range(len(attention_weights)):
        np_a = attention_weights[a][0].flatten()
        for w in range(len(np_a)):
            clean_attention_weights[a][w] = np_a[w]

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(clean_attention_weights, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + [vocab.itos[data[0][i]]
                               for i in range(seq_length)], rotation=90)
    ax.set_yticklabels([''] + [vocab.itos[targets[0][i]]
                               for i in range(seq_length)])

    # Show label at every tick
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    att = args.file_name + '/sentence_{}_at_epoch_{}'.format(count, epoch)

    plt.savefig('runs/'+att)
    plt.close()
