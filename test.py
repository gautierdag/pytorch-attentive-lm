from main import main

# Finding best hyper parameters for PTB
hidden_sizes = [250, 650, 1000]
lrs = [1, 0.1, 0.01, 0.001]
n_layers = [1, 2, 3]
attention = [False, True]

for h in hidden_sizes:
    for lr in lrs:
        for l in n_layers:
            for a in attention:
                params = ["--embedding-size", str(h), '--hidden-size',
                            str(h), '--lr', str(lr), '--n-layers', str(l),
                            '--batch-size', '20', '--epochs', '500', '--seed', '141']                
                if l > 1:
                    params.append('--rnn-dropout')
                    params.append('0.2')
                if a:
                    params.append('--no-attention')

                main(params)
                # if embedding size and hidden size same then we can tie weights - so try
                params.append('--tie-weights')
                main(params)
