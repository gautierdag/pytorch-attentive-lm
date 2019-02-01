# pytorch-attentive-lm


This repo is an implementation of an Attentive RNN model for the task of language modelling. 

Language modelling is done on both the PennTreeBank and Wikitext-02 datasets. The files are parsed such that each training example consists of one sentence from the corpus, padded to a max batch length of 35. Longer sentences are clipped. This is done in order to manage the attention and attend to only words in the sentence (before timestep t if at timestep t). 

The A-RNN-LM (Attention based Recurrent Neural Network for Language Modelling) was originally proposed in Coherent Dialogue with Attention-based Language Models (Hongyuan Mei et al. 2016, [link](https://arxiv.org/abs/1611.06997 "Coherent Dialogue with Attention-based Language Models")), and in Attentive Language Models (Salton et al. 2017, [link](https://www.semanticscholar.org/paper/Attentive-Language-Models-Salton-Ross/8a48edc093937a2f8ae665a4e1ecfa38972b234b "Attentive Language Models")). 

The model consists of running a traditional attention mechanism on the previous hidden states of the encoder RNN layer(s) to encode a context vector which is then combined with the last encoded hidden state in order to predict the next word in the sequence. 



## Installation and Usage

Dependencies:

  - `python=3.7`
  - `torch>=1.0.0`
  - `nltk`

Install all depedencies and run `python main.py`.

Multiple options for running are possible run `python main.py --help` for full list. 

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--lr LR] [--patience P]
               [--seed S] [--log-interval N] [--dataset [{wiki-02,ptb}]]
               [--embedding-size N] [--n-layers N] [--hidden-size N]
               [--input-dropout D] [--rnn-dropout D] [--decoder-dropout D]
               [--early-stopping-patience P] [--no-attention] [--tie-weights]
               [--use-hidden] [--file-name FILE_NAME]

PyTorch Attentive RNN Language Modeling

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --epochs N            number of epochs to train (default: 40)
  --lr LR               learning rate (default: 0.01)
  --patience P          patience (default: 10)
  --seed S              random seed (default: 123)
  --log-interval N      how many batches to wait before logging training
                        status (default 10)
  --dataset [{wiki-02,ptb}]
                        Select which dataset (default: ptb)
  --embedding-size N    embedding size for embedding layer (default: 65)
  --n-layers N          layer size for RNN encoder (default: 1)
  --hidden-size N       hidden size for RNN encoder (default: 65)
  --input-dropout D     input dropout (default: 0.5)
  --rnn-dropout D       rnn dropout (default: 0.0)
  --decoder-dropout D   decoder dropout (default: 0.5)
  --early-stopping-patience P
                        early stopping patience (default: 25)
  --no-attention        Disable attention (default: False
  --tie-weights         Tie embedding and decoder weights (default: False
  --use-hidden          Propagate hidden states over minibatches (default:
                        False
  --file-name FILE_NAME
                        Specific filename to save under (default: uses params
                        to generate
```
