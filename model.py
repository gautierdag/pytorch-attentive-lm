import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_size=65):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, hidden_size)
        self.attn_2 = nn.Linear(hidden_size, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x):

        attn_weights = F.softmax(self.attn_2(
            torch.tanh(self.attn_1(x))), dim=1)
        attn_applied = x * attn_weights

        return torch.sum(attn_applied, 1)


class LanguageModel(nn.Module):
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

        super(LanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.input_dropout = nn.Dropout(dropout_p_input)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, encoder_hidden_size, n_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout_p_encoder)

        self.attention = Attention(encoder_hidden_size, attention_hidden_size)

        self.decoder_dropout = nn.Dropout(dropout_p_decoder)
        self.decoder = nn.Linear(
            encoder_hidden_size+attention_hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if self.embedding_size != decoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, encoder embedding_size must be equal to hidden_size')
            self.decoder.weight = self.embedding.weight

        self.init_weights()

        self.U = nn.Linear(65, 65)
        self.W = nn.Linear(65, 65)
        self.b = nn.Linear(65, 1)

        self.Oz = nn.Linear(65, 65)
        self.Oh = nn.Linear(65, 65)
        self.Ov = nn.Linear(65, self.vocab_size)

    def forward(self, input, hidden):

        batch_size = input.shape[0]

        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)

        encoder_output, hidden = self.encoder(embedded, hidden)

        # Attention begin according to Attentive RNN-LM
        keys = self.W(encoder_output)  # 64x35x65
        queries = self.U(encoder_output)  # 64x35x65
        steps = queries.shape[1]
        # need to mask query 35*64*35*65
        mask = torch.ones(steps, steps).tril().view(
            steps, 1, steps, 1).expand(-1, batch_size, -1, self.encoder_hidden_size)

        Betas = self.b(torch.tanh(keys + queries*mask))
        alphas = F.softmax(Betas, dim=2)
        context_vectors = torch.sum(queries * alphas, dim=2).transpose(0, 1)

        # for t in range(steps):
        #     key = keys[:, t, :].clone()
        #     masked_queries = queries[:, :t+1, :].clone()

        #     Betas = self.b(torch.tanh(key.unsqueeze(1) + masked_queries))
        #     alphas = F.softmax(Betas, dim=1)
        #     z = torch.sum(masked_queries * alphas, dim=1)
        #     context_vectors.append(z)
        # # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        # context_vectors = torch.stack(
        #     context_vectors).view(batch_size, steps, -1)

        output = self.Ov(self.Oh(encoder_output) + self.Oz(context_vectors))
        # Attentive RNN-LM end

        return output, hidden

        # context_vector = self.attention(encoder_output)
        # output = self.decoder_dropout(encoder_output)
        # decoded = self.decoder(output.contiguous())

        # return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.encoder_hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.encoder_hidden_size))

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


# class PositionalLanguageModel(LanguageModel):
#     def __init__(self, vocab_size):
#         super(PositionalLanguageModel, self).__init__(vocab_size)

#         # n_building_blocks_mu = 4:   j, n, 1/n, j/n

#         # self.attn = nn.Linear(self.hidden_size * 2, 35)
#         # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.attn = Attention(self.encoder_module.hidden_size)

    # def forward(self, input, hidden):

    #     output, hidden = self.encoder_module(input, hidden=hidden)
    #     output = self.decoder_dropout(output)

    #     attended = self.attn(output.contiguous().view(-1, output.size(2)))

    #     decoded = self.decoder(attended)

    #     return decoded.view(output.size(0), output.size(1),
    #                         decoded.size(1)), hidden
