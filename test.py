from main import main

# Finding best hyper parameters for PTB
embedding_sizes = [250, 500, 1000]
hidden_sizes = [250, 500, 650, 800]
lrs = [0.1, 0.01, 0.001]
n_layers = [1, 2, 3]

for e in embedding_sizes:
    for h in hidden_sizes:
        for lr in lrs:
            for l in n_layers:
                main(["--embedding-size", str(e), '--hidden-size',
                      str(h), '--lr', str(lr), '--n-layers', str(l)])
