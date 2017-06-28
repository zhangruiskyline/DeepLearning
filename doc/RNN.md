<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Introduction](#introduction)
  - [Basic architecture](#basic-architecture)
  - [Model Analysis](#model-analysis)
- [LSTM](#lstm)
  - [LTSM structure](#ltsm-structure)
  - [GRU](#gru)
  - [Vanishing / Exploding Gradients](#vanishing--exploding-gradients)
    - [LSTM mitigate vanlish gradient](#lstm-mitigate-vanlish-gradient)
    - [LSTM activity function](#lstm-activity-function)
  - [LSTM and Sequence Model](#lstm-and-sequence-model)
    - [Model analysis](#model-analysis)
- [Seq-to-Seq Model](#seq-to-seq-model)
  - [Encoder/Decoder Model](#encoderdecoder-model)
- [Applications](#applications)
  - [Seq-to-Seq Application: Translation](#seq-to-seq-application-translation)
    - [Pre-Process](#pre-process)
    - [Build the model](#build-the-model)
  - [Seq-to-Seq Application: Chat Bot](#seq-to-seq-application-chat-bot)
  - [LSTM Application: Query Classification](#lstm-application-query-classification)
- [Attention model](#attention-model)
- [reference](#reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Introduction

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

## Basic architecture
![LSTM_cell](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/RNN.png)

The input x will be a sequence of words (just like the example printed above) and each x_t is a single word. But there’s one more thing: Because of how matrix multiplication works we can’t simply use a word index (like 36) as an input. Instead, we represent each word as a one-hot vector of size vocabulary_size. For example, the word with index 36 would be the vector of all 0’s and a 1 at position 36. So, each x_t will become a vector, and x will be a matrix, with each row representing a word. We’ll perform this transformation in our Neural Network code instead of doing it in the pre-processing. The output of our network o has a similar format. Each o_t is a vector of vocabulary_size elements, and each element represents the probability of that word being the next word in the sentence.

## Model Analysis
```math_def
\begin{aligned}  s_t &= \tanh(Ux_t + Ws_{t-1}) \\  o_t &= \mathrm{softmax}(Vs_t)  \end{aligned}  
```

# LSTM
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

## LTSM structure
![LSTM_cell](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LSTM_1.png)

* It has 4 times more parameters than RNN
* Mitigates vanishing gradient problem through gating

And the math model for each cell is
![LSTM_cell_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LSTM_2.png)

## GRU
GRU stands for Gated Recurrent Unit: similar idea as LSTM

* less parameters, as there is one gate less
* no "cell", only hidden vector ht is passed to next unit

In practice

* more recent, people tend to use LSTM more
* no systematic difference between the two

## Vanishing / Exploding Gradients

Passing through t time-steps, the resulting gradient is the product of many gradients and activations

* Gradients close to 0 will soon be 0
* Gradients larger than 1 will explode

### LSTM mitigate vanlish gradient

From the math, we can see forget gate can control the gradient vanlish
![LSTM_vanlish](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LSTM_vanlish.png)

![LSTM_vanlish_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LSTM_vanlish_2.png)
* Gradient clipping prevents gradient explosion
* Well chosen activation function is critical (tanh)



### LSTM activity function

All Gates inside LSTM cell needs to output between 0,1,  so we use Sigmoid and tanh function

The reason behind is

* The main problem in RNN is the vanishing gradient problem. Also, to keep the gradient in the linear region of the activation function, we need a function whose second derivative can sustain for a long range before going to zero.

* Sigmoid has output between [0,1] and tanh has output [-1,1]

* The gate recurrent connections use sigmoid, and the cell recurrent connections use tanh.

* Tanh having stronger gradients: since data is centered around 0, the derivatives are higher.

Below is a comparison among three different acitivation functions
![acitivity](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/activiation.png)


## LSTM and Sequence Model

word2vec or Glove use CBOW and skip gram to train, which ignore the order in word sequences. Using deep learning we need to deal with more

Assign a probability to a sequence of words, e.g:

```
p("I like cats")>p("I table cats")
p("I like cats")>p("like I cats")
```

The internal representation of the model has to better capture the meaning of a sequence than a simple Bag-of-Words.
Much more computationally expensive than Bag-of-Words models

Basically we need a sequence mode like
![seq-model](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq_model.png)

CBOW will be a special example of fixed sequence size.

For general model using RNN
 ![seq_RNN](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq_RNN.png)

 ![seq_RNN_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq_RNN_2.png)

  ![seq_RNN_3](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq_RNN_3.png)

### Model analysis

referring https://www.quora.com/How-are-inputs-fed-into-the-LSTM-RNN-network-in-mini-batch-method

* Variables involved:

Assume we have N data points (sentences), h hidden units (LSTM cells/blocks), b as mini-batch size, then it will take int(N/b)+1 epochs for the learner to go through all data points once. Let us assume that each sentence has length L. Also, suppose each word is represented by an embedding vector of dimensionality e. The input layer will have e linear units. These e linear units are connected to each of the h LSTM/RNN units in the hidden layer (assuming there is only one hidden layer).

* Feeding a sentence to a RNN:

In general, for any Recurrent Neural Network (RNN), there is a concept of time instances (steps) corresponding to a time-series or sequence. The words in a sentence are fed to the network one at a time. So, each of the L words in a sentence is fed to the network one by one (step by step). When one sentence has been fed completely, the activations of the units in the hidden layer are reset. Then, the next sentence is fed, and so on.

* Feeding a batch of b sentences to a RNN:

In step 1, first word of each of the b sentences (in a batch) is input in parallel. In step 2, second word of each of the b sentences is input in parallel. The parallelism is only for efficiency. Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. All the computations involving the words of all sentences in a batch at a given time step are done in parallel.

* Relation between L, e, h:

It is to be noted that, in general, there is no relation between L (length of a sentence), e (the dimension of the embedding space in which words are represented), and h (number of LSTM hidden units in the hidden layer).

# Seq-to-Seq Model

## Encoder/Decoder Model

Seq-to-Seq is first introduced in machine translation. Working with an input vocabulary of 160,000 English words, Sutskever et al. use an LSTM model to turn a sentence into 8000 real numbers. Those 8000 real numbers embody the meaning of the sentence well enough that a second LSTM can produce a French sentence from them. By this means English sentences can be translated into French. The overall process therefore takes a sequence of (English word) tokens as input, and produces another sequence of (French word) tokens as output.

A basic sequence-to-sequence model, as introduced in Cho et al., 2014 (pdf), consists of two recurrent neural networks (RNNs): an encoder that processes the input and a decoder that generates the output. This basic architecture is depicted below.

The idea is to use one LSTM to read the input sequence, one timestep at a time, to obtain large fixed-dimensional vector representation, and then to use another LSTM to extract the output sequence from that vector. The second LSTM is essentially a recurrent neural network language model except that it is conditioned on the input sequence. The LSTM’s ability to successfully learn on data with long range temporal dependencies makes it a natural choice for this application due to the considerable time lag between the inputs and their corresponding outputs.



![seq-to-seq](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq2seq.png)

Each box in the picture above represents a cell of the RNN, most commonly a GRU cell or an LSTM cell.

> In Summary: The Sequence to Sequence model (seq2seq) consists of two RNNs - an encoder and a decoder.

*  __Encoder__
 The encoder reads the input sequence, word by word and emits a context (a function of final hidden state of encoder)which would ideally capture the essence (semantic summary) of the input sequence.

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

![encoder](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/encoder.png)

* __Decoder__

Based on this context, the decoder generates the output sequence, one word at a time while looking at the context and the previous word during each timestep. This is a ridiculous oversimplification, but it gives you an idea of what happens in seq2seq.

![decoder](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/decoder.png)


* Goal
The context can be provided as the initial state of the decoder RNN or it can be connected to the hidden units at each time step. Now our objective is to jointly maximize the log probability of the output sequence conditioned on the input sequence.

# Applications
## Seq-to-Seq Application: Translation

http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/

![seq-to-seq example](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq2seq_example.png)

In the picture above, “Echt”, “Dicke” and “Kiste” words are fed into an encoder, and after a special signal (not shown) the decoder starts producing a translated sentence. The decoder keeps generating words until a special end of sentence token is produced. Here, the h vectors represent the internal state of the encoder.

* Sentence embedding

If you look closely, you can see that the decoder is supposed to generate a translation solely based on the last hidden state (h_3 above) from the encoder. This *h3* vector must encode everything we need to know about the source sentence. It must fully capture its meaning. In more technical terms, that vector is a __sentence embedding__.

* Reverse source

Still, it seems somewhat unreasonable to assume that we can encode all information about a potentially very long sentence into a single vector and then have the decoder produce a good translation based on only that. Let’s say your source sentence is 50 words long. The first word of the English translation is probably highly correlated with the first word of the source sentence. But that means decoder has to consider information from 50 steps ago, and that information needs to be somehow encoded in the vector.

In theory, architectures like LSTMs should be able to deal with this, but in practice long-range dependencies are still problematic. For example, researchers have found that reversing the source sequence (feeding it backwards into the encoder) produces significantly better results because it shortens the path from the decoder to the relevant parts of the encoder. Similarly, feeding an input sequence twice also seems to help a network to better memorize things.

### Pre-Process

* tokenize the source and target sequences;
* reverse the order of the source sequence;
* build the input sequence by concatenating the reversed source sequence and the target  sequence in original order using the GO token as a delimiter,
* build the output sequence by appending the EOS token to the source sequence.


To build the vocabulary we need to tokenize the sequences of symbols. For the digital number representation we use character level tokenization while whitespace-based word level tokenization will do for the French phrases:
```python
def tokenize(sentence, word_level=True):
    if word_level:
        return sentence.split()
    else:
        return [sentence[i:i + 1] for i in range(len(sentence))]
```

use this tokenization strategy to assign a unique integer token id to each possible token string found the traing set in each language

```python
def build_vocabulary(tokenized_sequences):
    rev_vocabulary = START_VOCAB[:]
    unique_tokens = set()
    for tokens in tokenized_sequences:
        unique_tokens.update(tokens)
    rev_vocabulary += sorted(unique_tokens)
    vocabulary = {}
    for i, token in enumerate(rev_vocabulary):
        vocabulary[token] = i
    return vocabulary, rev_vocabulary
tokenized_fr_train = [tokenize(s, word_level=True) for s in fr_train]
tokenized_num_train = [tokenize(s, word_level=False) for s in num_train]

fr_vocab, rev_fr_vocab = build_vocabulary(tokenized_fr_train)
num_vocab, rev_num_vocab = build_vocabulary(tokenized_num_train)

```

reverse the order
```python
def make_input_output(source_tokens, target_tokens, reverse_source=True):
    if reverse_source:
        source_tokens = source_tokens[::-1]
    input_tokens = source_tokens + [GO] + target_tokens
    output_tokens = target_tokens + [EOS]
    return input_tokens, output_tokens
```


> By using all of these, we can  apply the previous transformation to each pair of (source, target) sequene and use a shared vocabulary to store the results in numpy arrays of integer token ids, with padding on the left so that all input / output sequences have the same length:

```python
import numpy as np
max_length = 20  # found by introspection of our training set

def vectorize_corpus(source_sequences, target_sequences, shared_vocab,
                     word_level_source=True, word_level_target=True,
                     max_length=max_length):
    assert len(source_sequences) == len(target_sequences)
    n_sequences = len(source_sequences)
    source_ids = np.empty(shape=(n_sequences, max_length), dtype=np.int32)
    source_ids.fill(shared_vocab[PAD])
    target_ids = np.empty(shape=(n_sequences, max_length), dtype=np.int32)
    target_ids.fill(shared_vocab[PAD])
    numbered_pairs = zip(range(n_sequences), source_sequences, target_sequences)
    for i, source_seq, target_seq in numbered_pairs:
        source_tokens = tokenize(source_seq, word_level=word_level_source)
        target_tokens = tokenize(target_seq, word_level=word_level_target)

        in_tokens, out_tokens = make_input_output(source_tokens, target_tokens)

        in_token_ids = [shared_vocab.get(t, UNK) for t in in_tokens]
        source_ids[i, -len(in_token_ids):] = in_token_ids

        out_token_ids = [shared_vocab.get(t, UNK) for t in out_tokens]
        target_ids[i, -len(out_token_ids):] = out_token_ids
    return source_ids, target_ids

X_train, Y_train = vectorize_corpus(fr_train, num_train, shared_vocab,
                                    word_level_target=False)
```

```python
X_val, Y_val = vectorize_corpus(fr_val, num_val, shared_vocab,
                                word_level_target=False)
X_test, Y_test = vectorize_corpus(fr_test, num_test, shared_vocab,
                                  word_level_target=False)
```

### Build the model

* Start with an Embedding layer;
* Add a single GRU layer: the GRU layer should yield a sequence of output vectors, one at each timestep;
* Add a Dense layer to adapt the ouput dimension of the GRU layer to the dimension of the output vocabulary;
* Don't forget to insert some Dropout layer(s), especially after the Embedding layer.

Some notes:

* The output dimension of the Embedding layer should be smaller than usual be cause we have small vocabulary size;
* The dimension of the GRU should be larger to give the Seq2Seq model enough "working memory" to memorize the full input sequence before decoding it;
* The model should output a shape [batch, sequence_length, vocab_size].

```python
from keras.models import Sequential
from keras.layers import Embedding, Dropout, GRU, Dense

vocab_size = len(shared_vocab)
simple_seq2seq = Sequential()
simple_seq2seq.add(Embedding(vocab_size, 32, input_length=max_length))
simple_seq2seq.add(Dropout(0.2))
simple_seq2seq.add(GRU(256, return_sequences=True))
simple_seq2seq.add(Dense(vocab_size, activation='softmax'))

# Here we use the sparse_categorical_crossentropy loss to be able to pass
# integer-coded output for the token ids without having to convert to one-hot
# codes
simple_seq2seq.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


best_model_fname = "simple_seq2seq_checkpoint.h5"
best_model_cb = ModelCheckpoint(best_model_fname, monitor='val_loss',
                                save_best_only=True, verbose=1)
```

## Seq-to-Seq Application: Chat Bot

![seq-to-seq chatbot](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/seq2seq_chatbot.png)

## LSTM Application: Query Classification
http://campuspress.yale.edu/yw355/deep_learning/

# Attention model
The above translate mode, every input has to be encoded into a fixed-size state vector, as that is the only thing passed to the decoder(The final sentence embedding). To allow the decoder more direct access to the input, an attention mechanism was introduced.

With an attention mechanism we no longer try encode the full source sentence into a fixed-length vector. Rather, we allow the decoder to “attend” to different parts of the source sentence at each step of the output generation.

The idea is to let every step of an RNN pick information to look at from some larger collection of information.

Take machine Translation example, Each time the decoder RNN produces a word, it determines the contribution of each hidden states to take as input(instead of only the last one hidden state). The contribution computed using a __softmax__: this means that attention weights a_j are computed such that __sum a_j = 1__

![attention](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/attention.png)


This process can be seen as an alignment, because the network usually learns to focus on a single input word each time it produces an output word. This means that most of the attention weights are 0 (black) while a single one is activated (white).he image below shows the attention weights during the translation process, which reveals the alignment and makes it possible to interpret what the network has learnt (and this is usually a problem with RNNs!)

![attention_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/attention_2.png)

# reference

http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
https://blog.heuritech.com/2016/01/20/attention-mechanism/
https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
