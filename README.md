# pytorch-attentive-lm


This repo is an implementation of an Attentive RNN model for the task of language modelling. 

Language modelling is done on both the PennTreeBank and Wikitext-02 datasets. The files are parsed such that each training example consists of one sentence from the corpus, padded to a max batch length of 35. Longer sentences are clipped. This is done in order to manage the attention and attend to only words in the sentence (before timestep t if at timestep t). 

The A-RNN-LM (Attention based Recurrent Neural Network for Language Modelling) was originally proposed in Coherent Dialogue with Attention-based Language Models (Hongyuan Mei et al. 2016, [link](https://arxiv.org/abs/1611.06997 "Coherent Dialogue with Attention-based Language Models")), and in Attentive Language Models (Salton et al. 2017, [link](https://www.semanticscholar.org/paper/Attentive-Language-Models-Salton-Ross/8a48edc093937a2f8ae665a4e1ecfa38972b234b "Attentive Language Models")). 

![Architecture](https://i.imgur.com/Y0rlTCg.png)

The model consists of running a traditional attention mechanism on the previous hidden states of the encoder RNN layer(s) to encode a context vector which is then combined with the last encoded hidden state in order to predict the next word in the sequence. 


## Installation and Usage

Dependencies:

  - `python=3.7`
  - `torch>=1.0.0`
  - `nltk`
  - `matplotlib`
  - `tensorboardX`

Install all depedencies and run `python main.py`.

The datasets will be downloaded and pre-processed automatically. 

Multiple options for running are possible run `python main.py --help` for full list. 

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--lr LR] [--patience P]
               [--seed S] [--log-interval N] [--dataset [{wiki-02,ptb}]]
               [--embedding-size N] [--n-layers N] [--hidden-size N]
               [--positioning-embedding N] [--input-dropout D]
               [--rnn-dropout D] [--decoder-dropout D] [--clip N]
               [--optim [{sgd,adam,asgd}]] [--salton-lr-schedule]
               [--early-stopping-patience P] [--attention]
               [--no-positional-attention] [--tie-weights]
               [--file-name FILE_NAME] [--parallel]

PyTorch Attentive RNN Language Modeling

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --epochs N            number of epochs to train (default: 40)
  --lr LR               learning rate (default: 30.0)
  --patience P          patience for lr decrease (default: 5)
  --seed S              random seed (default: 123)
  --log-interval N      how many batches to wait before logging training
                        status (default 10)
  --dataset [{wiki-02,ptb}]
                        Select which dataset (default: ptb)
  --embedding-size N    embedding size for embedding layer (default: 20)
  --n-layers N          layer size for RNN encoder (default: 1)
  --hidden-size N       hidden size for RNN encoder (default: 20)
  --positioning-embedding N
                        hidden size for positioning generator (default: 20)
  --input-dropout D     input dropout (default: 0.5)
  --rnn-dropout D       rnn dropout (default: 0.0)
  --decoder-dropout D   decoder dropout (default: 0.5)
  --clip N              value at which to clip the norm of gradients (default:
                        0.25)
  --optim [{sgd,adam,asgd}]
                        Select which optimizer (default: sgd)
  --salton-lr-schedule  Enables same training schedule as Salton et al. 2017
                        (default: False)
  --early-stopping-patience P
                        early stopping patience (default: 25)
  --attention           Enable standard attention (default: False)
  --no-positional-attention
                        Disable positional attention (default: False)
  --tie-weights         Tie embedding and decoder weights (default: False)
  --file-name FILE_NAME
                        Specific filename to save under (default: uses params
                        to generate)
  --parallel            Enable using GPUs in parallel (default: False)
```

## Results

### Results on PTB:
| Model| Number of parameters | Validation Perplexity | Test Perplexity  |
| ------------- | :-------------: | :-------------:| :---------:|
| LSTM Baseline (Merity et al., 2017)| 7.86M | **66.77** | **64.96**
| Attentive LM (Salton et al. 2017)| 7.06M  |  79.09 | 76.56
| Positional Attentive LM | **6.9M** |  72.69 | 70.92


### Results on Wikitext-02:
| Model| Number of parameters | Validation Perplexity | Test Perplexity  |
| ------------- | :-------------: | :-------------:| :---------:|
| LSTM Baseline (Merity et al., 2017)| 7.86M | **72.43** | **68.50**
| Attentive LM (Salton et al. 2017)|  7.06M  |  78.43 | 74.37
| Positional Attentive LM | **6.9M** |  74.39| 70.73


### Reproducing results: 
You can rerun all the models which generated the tables above by simply running:

`python test.py`

However please note that some of these models take upwards of 8hours to converge on a single 1080 GPU, so the total run-time of the experiment could be approximately 2 days. 

Multi-GPU support is disabled by default as it was shown to have a negative impact on results. On top of that since batches are small in practice it is not actually much faster since a lot of time is spent sending the tensors to the respective GPUs.

### Comparing attentions 

Here are shown side by side comparaisons of the two attention distributions on an example: 

![Standard Attention](https://i.imgur.com/15rcNWc.png "Standard Attention") ![Positional Attention](https://i.imgur.com/q5CiIjV.png "Positional Attention")

The words in the X-axis are the inputs at each time step and the words in the Y-axis are the targets. Both models were trained on the Wikitext-02 dataset until convergence. 
