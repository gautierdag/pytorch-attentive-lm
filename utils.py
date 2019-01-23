def generate_filename(args):
    """
    Generates a filename from arguments passed
    Used for saving model and runs
    """
    n = args.dataset+'_n_layers_'+str(args.n_layers)+'_hidden_size_' + str(args.hidden_size) + \
         '_lr_' + str(args.lr)

    if args.tie_weights:
        n += '_tied_weights'

    if not args.no_attention:
        n += '_no_attention'

    return n
