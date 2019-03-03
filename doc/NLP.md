<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Tradirional NLP Method and Fudamental](#tradirional-nlp-method-and-fudamental)
  - [Transformer Model](#transformer-model)
    - [Transformer Model with Code](#transformer-model-with-code)
  - [BERT](#bert)
    - [Pretraining](#pretraining)
      - [Word2Vec](#word2vec)
      - [Embedding from Language Models](#embedding-from-language-models)
    - [Architecture](#architecture)
    - [Blog Post](#blog-post)
- [reference](#reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Tradirional NLP Method and Fudamental

Pleasew refer my other Repo: https://github.com/zhangruiskyline/NLP_DeepLearning

## Transformer Model

https://jalammar.github.io/illustrated-transformer/

In contrast, the Transformer only performs a small, constant number of steps (chosen empirically). In each step, it applies a self-attention mechanism which directly models relationships between all words in a sentence, regardless of their respective position.

### Transformer Model with Code
http://nlp.seas.harvard.edu/2018/04/03/attention.html

## BERT

https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

* Challenge

One of the biggest challenges in natural language processing (NLP) is the shortage of training data. Because NLP is a diversified field with many distinct tasks, most task-specific datasets contain only a few thousand or a few hundred thousand human-labeled training examples. However, modern deep learning-based NLP models see benefits from much larger amounts of data, improving when trained on millions, or billions, of annotated training examples. 

* PreTrain data

researchers have developed a variety of techniques for training general purpose language representation models using the enormous amount of unannotated text on the web (known as pre-training). 

* So it is about pretrain + fine tune

### Pretraining 

#### Word2Vec

> Word2vec used as input for other NN network

![W2V_Pretrain](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/W2V_pretrain.jpg)


> Drawback of Word2Vec

#### Embedding from Language Models

Understand Context: Word2vec just uses fixed vector regardless of conetxt. But ELMO first uses pretrain data, and then adjust word's vector based on context. 

ELMO uses two stage training processes:

* Stage 1 uses pretrained word embedding .

* Stage 2: For downstream application, use pretrained word embedding as new features

> The network Architecture is as below

![ELMO](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ELMO.jpg)

Every encoder is a 2 layer Bi-LSTM. after the training, every new sentence will have 3 embedding

1. Word embedding
2. first B-LSTM layer word position embedding, so this layer will have sentence context 
3. second B-LSTM layer word position embedding, so this layer will have semantic context 

> How to use ELMO

The application  of pretrained ELMO can be used in down stream application. For example, Q&A as before 

![ELMO_application](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ELMO_application.jpg)

### Architecture

Pre-trained representations can either be context-free or contextual, and contextual representations can further be unidirectional or bidirectional. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary. For example, the word “bank” would have the same context-free representation in “bank account” and “bank of the river.” Contextual models instead generate a representation of each word that is based on the other words in the sentence. For example, in the sentence “I accessed the bank account,” a unidirectional contextual model would represent “bank” based on “I accessed the” but not “account.” However, BERT represents “bank” using both its previous and next context — “I accessed the ... account” — starting from the very bottom of a deep neural network, making it deeply bidirectional.


![BERT](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/BERT.png)

> The Strength of Bidirectionality

it is not possible to train bidirectional models by simply conditioning each word on its previous and next words, since this would allow the word that’s being predicted to indirectly “see itself” in a multi-layer model. 

To solve this problem, we use the straightforward technique of masking out some of the words in the input and then condition each word bidirectionally to predict the masked words.

BERT also learns to model relationships between sentences by pre-training on a very simple task that can be generated from any text corpus: Given two sentences A and B, is B the actual next sentence that comes after A in the corpus, or just a random sentence?


### Blog Post

https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3




# reference

http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
https://blog.heuritech.com/2016/01/20/attention-mechanism/
https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
