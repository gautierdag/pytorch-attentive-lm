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


# Finding baselines
params_attentive_lm = ["--embedding-size", '650', '--hidden-size', '650', '--n-layers', '2',
                       '--batch-size', '32', '--epochs', '500', '--seed', '1701',
                       '--log-interval', '200', '--early-stopping-patience', '10', '--lr', '1.0',
                       '--tie-weights', '--file-name', 'attentive_lm_baseline_true_learning']
main(params_attentive_lm)
