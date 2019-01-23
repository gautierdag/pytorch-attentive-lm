import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x):

        # attn_weights = F.softmax(self.attn_2(
        #     torch.tanh(self.attn_1(x))), dim=1)

        # attn_applied = x * attn_weights
        # return torch.sum(attn_applied, 1)

        attn_weights = self.attn_2(torch.tanh(self.attn_1(x)))
        return attn_weights


class AttentiveRNNLanguageModel(nn.Module):
    """
    Implements a language model

    """

    def __init__(self, vocab_size,
                 embedding_size=65,
                 encoder_hidden_size=65,
                 n_layers=1,
                 max_len=35,
                 bidirectional=False,
                 attention_hidden_size=65,
                 decoder_hidden_size=65,
                 tie_weights=False,
                 dropout_p_input=0.5,
                 dropout_p_encoder=0.0,
                 dropout_p_decoder=0.5,
                 attention=False):

        super(AttentiveRNNLanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.input_dropout = nn.Dropout(dropout_p_input)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, encoder_hidden_size, n_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout_p_encoder)

        self.attention = Attention(encoder_hidden_size)

        self.decoder_dropout = nn.Dropout(dropout_p_decoder)
        self.decoder = nn.Linear(encoder_hidden_size*2, vocab_size)
        self.init_weights()

    def forward(self, input):

        batch_size = input.shape[0]
        sequence_length = input.shape[1]

        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)

        encoder_output, _ = self.encoder(embedded)

        self_attention_scores = self.attention(encoder_output)

        context_vectors = []
        for t in range(sequence_length):
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t+1, :].clone(), dim=1)
            context_vectors.append(
                torch.sum(weighted_attention_scores*encoder_output[:, :t+1, :].clone(), dim=1))

        context_vectors = torch.stack(context_vectors).transpose(0, 1)
        combined_endoding = torch.cat((context_vectors, encoder_output), dim=2)
        output = self.decoder_dropout(combined_endoding)
        decoded = self.decoder(output.contiguous())

        return decoded

    def flatten_parameters(self):
        """
        Flatten parameters of all reccurrent components in the model.
        """
        self.encoder.flatten_parameters()

    def init_weights(self):
        """
        Standard weight initialization
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(
            -initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
