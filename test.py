from main import main

# # Finding baselines
# params_merity = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '3',
#                  '--batch-size', '20', '--epochs', '500', '--seed', '123',
#                  '--log-interval', '200', '--patience', '5', '--lr', '30.0',
#                  '--rnn-dropout', '0.25', '--tie-weights', '--input-dropout',
#                  '0.4', '--file-name', 'merity', '--no-positional-attention']
# main(params_merity)


# params_attentive_lm = ["--embedding-size", '650', '--hidden-size', '650', '--n-layers', '2',
#                        '--batch-size', '32', '--epochs', '500', '--seed', '123',
#                        '--log-interval', '200', '--patience', '5', '--lr', '30.0',
#                        '--rnn-dropout', '0.2', '--tie-weights', '--file-name',
#                        'salton', '--attention', '--no-positional-attention']
# main(params_attentive_lm)


# positional_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
#                  '--batch-size', '32', '--epochs', '500', '--seed', '123', '--rnn-dropout', '0.2',
#                  '--log-interval', '200', '--lr', '30.0','--tie-weights', '--file-name',
#                   'positional', '--patience', '5']
# main(positional_lm)

# params_merity = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '3',
#                  '--batch-size', '20', '--epochs', '750', '--seed', '123',
#                  '--log-interval', '200', '--patience', '5', '--lr', '30.0',
#                  '--rnn-dropout', '0.2', '--tie-weights',
#                  '--file-name', 'wiki_merity', '--no-positional-attention', '--dataset', 'wiki-02']
# main(params_merity)

params_attentive_lm = ["--embedding-size", '1000', '--hidden-size', '1000', '--n-layers', '2',
                       '--batch-size', '32', '--epochs', '500', '--seed', '123',
                       '--log-interval', '200', '--patience', '6', '--lr', '30.0',
                       '--rnn-dropout', '0.2', '--tie-weights', '--file-name', 'wiki_salton',
                       '--attention', '--no-positional-attention', '--dataset', 'wiki-02']
main(params_attentive_lm)


positional_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                 '--batch-size', '32', '--epochs', '500', '--seed', '123', '--rnn-dropout', '0.2',
                 '--log-interval', '200', '--lr', '30.0', '--tie-weights', '--file-name',
                 'positional_pos_gen_100', '--patience', '5', '--positioning-embedding', '100']
main(positional_lm)
