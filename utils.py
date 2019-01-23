def generate_filename(args):
    """
    Generates a filename from arguments passed
    Used for saving model and runs
    """
    n = args.dataset+'_embed_size_'+str(args.embedding_size) + \
        '_n_layers_'+str(args.n_layers)+'_hidden_size_' + str(args.hidden_size) + \
        '_i_d_' + str(args.input_dropout)+'_e_d_'+str(args.rnn_dropout) + \
        '_o_d_' + str(args.decoder_dropout) + '_lr_' + str(args.lr)

    if args.lr_decay != 0.1:
        n += '_lr_decay_' + str(args.lr_decay)

    if args.patience != 30:
        n += '_patience_' + str(args.patience)

    if not args.no_attention:
        n += '_no_attention'

    return n
