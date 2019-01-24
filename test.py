from main import main

# Finding best hyper parameters for PTB
hidden_sizes = [650, 1000]
n_layers = [1, 2]
attention = [False, True]

for h in hidden_sizes:
        for l in n_layers:
            if h == 650 and l == 1:
                continue
            for a in attention:
                params = ["--embedding-size", str(h), '--hidden-size',
                            str(h), '--n-layers', str(l),
                            '--batch-size', '20', '--epochs', '100', '--seed', '141',
                            '--log-interval', '1000', '--patience', '5']
                if l > 1:
                    params.append('--rnn-dropout')
                    params.append('0.2')
                if a:
                    params.append('--no-attention')
                print(params)
                main(params)
                # if embedding size and hidden size same then we can tie weights - so try
                params.append('--tie-weights')
                main(params)
