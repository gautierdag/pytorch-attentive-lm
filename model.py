import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        self.concatenation_layer = nn.Linear(feature_dim * 2, feature_dim)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x):
        # Input x is encoder output
        sequence_length = x.shape[1]

        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))

        # Attend for each time step using the previous context
        context_vectors = []
        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t + 1, :].clone(), dim=1))

        context_vectors = torch.stack(context_vectors).transpose(0, 1)
        combined_encoding = torch.cat(
            (context_vectors, x), dim=2)

        # concatenation layer
        combined_output = torch.tanh(
            self.concatenation_layer(combined_encoding))

        return combined_output


class PositionalAttention(nn.Module):
    def __init__(self,
                 feature_dim,
                 positioning_embedding=20,
                 num_building_blocks=4,
                 max_len=35):
        super(PositionalAttention, self).__init__()
        self.num_building_blocks = num_building_blocks

        self.positioning_generator = nn.LSTM(
            feature_dim, positioning_embedding, batch_first=True)

        self.sigma_generator = nn.Linear(positioning_embedding, 1)
        self.mu_generator = nn.Linear(
            positioning_embedding, num_building_blocks)

    def forward(self, x):

        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        positioning_weights, _ = self.positioning_generator(x)
        mu_weights = self.mu_generator(positioning_weights)
        sigma_weights = self.sigma_generator(positioning_weights)

        prev_mu = torch.zeros(batch_size)
        building_blocks = torch.ones((batch_size, self.num_building_blocks))
        # Attend for each time step using the previous context
        position_vectors = []  # Which positions to attend to
        for j in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states

            building_blocks[:, 0] = prev_mu
            building_blocks[:, 1] = 1/sequence_length
            building_blocks[:, 2] = (j+1)/sequence_length

            mu = mu_weights[:, j, :].clone() @ building_blocks
            prev_mu = mu
            sigma = sigma_weights[:, j, :]

            gaussian_weighted_attention = torch.zeros(
                (50, 50)).normal_(mean=mu, std=sigma)

            current_encoded_sequence = x[:, :j+1, :].clone()
            applied_positional_attention = current_encoded_sequence * gaussian_weighted_attention
            position_vectors.append(
                torch.sum(applied_positional_attention, dim=1))

        context_vectors = torch.stack(position_vectors).transpose(0, 1)
        combined_encoding = torch.cat((context_vectors, x), dim=2)

        return combined_encoding


class AttentiveRNNLanguageModel(nn.Module):
    """
    Implements an Attentive Language Model according to http://www.aclweb.org/anthology/I17-1045

    """

    def __init__(self, vocab_size,
                 embedding_size=65,
                 hidden_size=65,
                 n_layers=1,
                 dropout_p_input=0.5,
                 dropout_p_encoder=0.0,
                 dropout_p_decoder=0.5,
                 attention=False,
                 positional_attention=True,
                 tie_weights=True,
                 use_hidden=False):

        super(AttentiveRNNLanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.attention = attention
        self.positional_attention = positional_attention
        self.use_hidden = use_hidden

        self.input_dropout = nn.Dropout(dropout_p_input)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size,
                               n_layers, batch_first=True,
                               dropout=dropout_p_encoder)
        if self.attention:
            self.attention_score_module = Attention(hidden_size)
        if self.positional_attention:
            self.position_score_module = PositionalAttention(hidden_size)

        if self.attention and self.positional_attention:
            raise NotImplementedError(
                "Attention and Positional Attention cannot be both activated")

        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.decoder_dropout = nn.Dropout(dropout_p_decoder)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if self.embedding_size != hidden_size:
                raise ValueError(
                    'When using the tied flag, encoder embedding_size must be equal to hidden_size')
            self.decoder.weight = self.embedding.weight

        self.init_weights()

    def forward(self, input, hidden):

        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)

        if self.use_hidden:
            encoder_output, hidden = self.encoder(embedded, hidden)
        else:
            encoder_output, hidden = self.encoder(embedded)

        if self.attention:
            encoder_output = self.attention_score_module(encoder_output)
        if self.positional_attention:
            encoder_output = self.position_score_module(encoder_output)

        output = self.decoder_dropout(encoder_output)
        decoded = self.decoder(output.contiguous())

        return decoded, hidden

    def flatten_parameters(self):
        """
        Flatten parameters of all reccurrent components in the model.
        """
        self.encoder.flatten_parameters()

    def init_weights(self, init_range=0.1):
        """
        Standard weight initialization
        """
        self.embedding.weight.data.uniform_(
            -init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        # initialize hidden state for RNN layer
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
