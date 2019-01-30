from main import main

# # Finding baselines
# params_attentive_lm = ["--embedding-size", '650', '--hidden-size', '650', '--n-layers', '2',
#                        '--batch-size', '32', '--epochs', '500', '--seed', '141',
#                        '--log-interval', '200', '--patience', '5', '--lr', '0.01',
#                        '--rnn-dropout', '0.2', '--tie-weights', '--file-name',
#                        'attentive_lm_baseline_use_hidden', '--use-hidden']
# main(params_attentive_lm)

# params_merity = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '3',
#                  '--batch-size', '20', '--epochs', '500', '--seed', '141',
#                  '--log-interval', '200', '--patience', '5', '--lr', '0.01',
#                  '--rnn-dropout', '0.25', '--tie-weights', '--input-dropout',
#                  '0.4', '--file-name', 'lstm_baseline_use_hidden', '--no-attention', '--use-hidden']
# main(params_merity)


positional_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                 '--batch-size', '32', '--epochs', '500', '--seed', '123',
                 '--log-interval', '200', '--early-stopping-patience', '10', '--lr', '0.1',
                 '--tie-weights', '--file-name', 'positional_test_adam',
                 '--patience', '5', '--optim', 'adam']
main(positional_lm)
positional_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                 '--batch-size', '32', '--epochs', '500', '--seed', '123',
                 '--log-interval', '200', '--early-stopping-patience', '10', '--lr', '0.1',
                 '--tie-weights', '--file-name', 'positional_test_wiki02_adam',
                 '--patience', '5', '--optim', 'adam', '--dataset', 'wiki-02']
main(positional_lm)


attention_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                '--batch-size', '32', '--epochs', '500', '--seed', '123',
                '--log-interval', '200', '--early-stopping-patience', '10', '--lr', '0.1',
                '--tie-weights', '--file-name', 'attention_test_adam', '--patience', '5', '--optim',
                'adam', '--attention', '--no-positional-attention']
main(attention_lm)

attention_lm = ["--embedding-size", '400', '--hidden-size', '400', '--n-layers', '2',
                '--batch-size', '32', '--epochs', '500', '--seed', '123',
                '--log-interval', '200', '--early-stopping-patience', '10', '--lr', '0.1',
                '--tie-weights', '--file-name', 'attention_test_wiki02_adam', '--patience', '5', '--optim',
                'adam', '--attention', '--no-positional-attention', '--dataset', 'wiki-02']
main(attention_lm)
