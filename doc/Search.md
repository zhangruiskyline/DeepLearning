<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Overview](#overview)
  - [Query/Document Match](#querydocument-match)
  - [Machine Learning based search system](#machine-learning-based-search-system)
- [Core Search/Rank Algorithm](#core-searchrank-algorithm)
  - [TF-IDF](#tf-idf)
  - [BM25](#bm25)
  - [Language Model](#language-model)
- [Query-Document Understanding](#query-document-understanding)
  - [Query Keywords understanding](#query-keywords-understanding)
    - [Classification](#classification)
    - [Query Parse](#query-parse)
    - [Query Expansion](#query-expansion)
  - [Documentation understanding](#documentation-understanding)
- [Machine Learn to Ranking](#machine-learn-to-ranking)
  - [Pointwise Learning to Rank](#pointwise-learning-to-rank)
  - [PairWise Learning to Rank](#pairwise-learning-to-rank)
  - [Listwise Learning to Rank](#listwise-learning-to-rank)
  - [RankSVM](#ranksvm)
  - [GBDT: Gradient Boost Decision Tree](#gbdt-gradient-boost-decision-tree)
  - [LambdaMART](#lambdamart)
- [Two stage: Selection and Ranking](#two-stage-selection-and-ranking)
  - [Top K Selection](#top-k-selection)
    - [Mutiple recall](#mutiple-recall)
  - [Classical Recall Algorithms](#classical-recall-algorithms)
  - [Re-Ranking](#re-ranking)
  - [balance between recall/precision](#balance-between-recallprecision)
  - [High frequency vs Long tail](#high-frequency-vs-long-tail)
- [Deep Learning Based Ranking](#deep-learning-based-ranking)
  - [Emebedding representation](#emebedding-representation)
  - [CNN based semantic model](#cnn-based-semantic-model)
  - [Learning local representation](#learning-local-representation)
- [Evaluation](#evaluation)
  - [Offline](#offline)
    - [ROC](#roc)
    - [F1](#f1)
    - [NDCG(Normalized Discounted Cumulative Gain)](#ndcgnormalized-discounted-cumulative-gain)
  - [Online](#online)
- [CTR(Click through rate)](#ctrclick-through-rate)
  - [Unique in CTR](#unique-in-ctr)
  - [Classic Method](#classic-method)
    - [Logistic regression](#logistic-regression)
    - [Factor Machine](#factor-machine)
    - [GBDT](#gbdt)
  - [Deep Learning on CTR](#deep-learning-on-ctr)
    - [Deep Learning feature set](#deep-learning-feature-set)
    - [Handle the challenge](#handle-the-challenge)
    - [one hot to dense](#one-hot-to-dense)
    - [General DNN network Architecture](#general-dnn-network-architecture)
    - [Advanced architecture based on general architecture](#advanced-architecture-based-on-general-architecture)
      - [Parallel](#parallel)
      - [Serialization](#serialization)
  - [DNN model training and optimization](#dnn-model-training-and-optimization)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->






# Overview

## Query/Document Match

This is the main stream for search algorithm since 1950s. The key steps for this work is to

* Build the Inverted Index

for each query, build a list of documents that have that keywords, not necessary to store all documents, just its index

* Core Ranking

Ranking documents with query, like TF-IDF, BM25

## Machine Learning based search system

Leverage machine learning system to build the models: select features, build the model, and validation. compared with query/documentation match, machine learning system can leverage more multi modal data, like text, image . etc..



# Core Search/Rank Algorithm

## TF-IDF

## BM25

## Language Model

# Query-Document Understanding

To understand the Query and Documentation

## Query Keywords understanding


### Classification

* Classification could be done via

 > Multiclass，Single-Label，Hard Classification
 > Multiclass，Multi-Label，Hard Classification

* Features used

Bag of Words, N-diagram

> N-Grams:

The good part of using N-Grams as it can expand the feature space and bring more features. but the cons is feature dimension could be too large, and very sparse.

* Algorithms

Logistic regression, SVM, Naive Bayes can all be used.

### Query Parse

We need to unstand what is intent for a certain query, most common way is to construct a __Inverted Index__. we can abstract some targeted documenattions from all documentation, and do re ranking later.

Referring to [Query Segmentation revisited] (https://www.cse.iitb.ac.in/~soumen/doc/www2013/QirWoo/HagenPSB2011QuerySegmentRevisit.pdf)

Three main techniques we can used

* N-Grams

* Mutual Information

* Conditional random Field

### Query Expansion

Query Expansion can improve the recall rate. For example, some one search for "Iphone10 repair", if we can expand the query to all "iphone repair" we may get more accurate results. But sometimes it may lose some accuracy(suppose iphone10 has some specific repair)

Another application for query Expansion is to process the equal meaning or Acronym words. Like "Donald Trump" to "POTUS"

* Method 1 is to build a graph relationship between search query and keywords.

* Method 2 is to use word Embeddings type of Embedding space. For each query, we build up a representation, use contextual



## Documentation understanding

# Machine Learn to Ranking

Learning to rank or machine-learned ranking (MLR) is the application of machine learning, typically supervised, semi-supervised or reinforcement learning, in the construction of ranking models for information retrieval systems.Training data consists of lists of items with some partial order specified between items in each list. This order is typically induced by giving a numerical or ordinal score or a binary judgment (e.g. "relevant" or "not relevant") for each item

Compared with unsupervised learning like TF-IDF and BM25. It will leverage the supervised learning scheme.

![Machine Learning rank](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rank_ML.png)


## Pointwise Learning to Rank

Turn the ranking to Binary Classification problem. The Training model will be pair of __{keywords, Documentation}__. The results is binary  either relevant or not.

> Regarding the Evaluation, we can evaluate the results from precision and recall.

* Ranking precision: Out of all Documentation judged as relevant, how much of them are marked as relevant.
* Ranking Recall: Out of all relevant Documentation, how much of them have been marked as relevant

> Different than other learning to rank problem, for each keywords, will only evaluate Top K Documentation.  

In reality, K will be like 3,5,10, but the total documentation size could be very large. So how to choose the K documentation is very important

## PairWise Learning to Rank

The drawback of Pointwise Learning to Rank is that the goal is very vague. The most important is not asses relevance for a certain documentation, rather it is assess the relevant ranking for a group documentation.

So we change the __{keywords, Documentation}__ to __{keywords, Documentation/documentation}__. Assume we have 3 documentation, The perfect ranking B>C>A, then we expect to learn the PairWise ranking like B>C and C>A.

There are several important assumptions here:

* There is perfect ranking for documentations. We can get it from 5 labels system, or some online metrics like CTR. but it may not be available. Like in e-commence website , some one search for "Lord of Ring", they may want a book, or movie, so very hard to get a perfect ranking.

* We assume the difference between documentations come from the "features" difference between them. so we can learn their feature difference. like RankSVM, it is still based on SVM, but training data becomes the pair of documentation rather than single documentation

In reality, computation overhead could be high since we need to check every pair of documentation

## Listwise Learning to Rank

The idea of Listwise learning to rank is trying to re construct the perfect ranking based on known ranking, it will directly optimize the NDCG, and compare with perfect known ranking, get the diff and learn to optimize.

NDCG is none continuous and none Differentiable, so optimization will be hard. Algorithm like ListMLE and ListNet could be used.

## RankSVM

> convert the ranking problem into a machine learning problem and use SVM to solve

* Collect a set of features, X, like documentation information, query/documentation similarity, query keywords. etc , and label, Y, to indicate its correlation

* The ranking algorithm should work like this: X1, X2 two sets of feature lists. and its correposnding Y1=3, Y2=5, then the ranking algorithm should rank documentation with X2 higher than documentation with X1

* So we need to learning a W(linear transition),

```
X2*W > X1*W
```
In fact , the hyperspace W build should make the distance between two set maximum

> However, Ranking SVM has problem as

* Complexity is the __O(N^2)__, where N is the data set number, since we need to build the pairwise feature/label set

## GBDT: Gradient Boost Decision Tree

The idea of boost to use a set of weak classifier to build a strong classifier.

## LambdaMART

* RankNet

Ranknet requires to define a loss function to learn rank algorithm, which they choose Logistic Regression.

Ranknet can learn the pairwise between two documentation, but as we indicated before the pairwise documentation can be represent exact ranking, especially this kind of algorithm lack the optimization on metrics such as NDCG.

For example,  for certain query keywords, we have 10 documentations as top K. and two of them have similarity 5 and 3, in ranking, similarity 5 ranked as 4th position while similarity 3 ranks 7th position,  so ranknet could be more optimize to increase 7th rank higher. so it will de prioritize un correlated, and increase the score(lower loss function).

But in fact, in NDCG, rank similarity 5 from 4th to higher may have larger impact.

* LambdaMART

During RankNet training procedure, you don’t need the costs, only need the gradients (λ) of the cost with respect to the model score. You can think of these gradients as little arrows attached to each document in the ranked list, indicating the direction we’d like those documents to move.

![LambadaMART](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lambdamart.png)

# Two stage: Selection and Ranking

The challenge for current search system is most model like Tree based (GBDT) and Deep learning has high computation complexity. but in inference stage for online ranking. the delay requirement is very tight(<1s), so it would be very hard to rank all related documentation, sometimes it would be >1m.

The idea of two stage recall can be shown as below

![2stagerecall](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/2stagerecall.jpg)

## Top K Selection

We need to pick top K(hundreds level) from all documentations . the model could be simple. mostly widely used is inverted index. And another way is the WAND operator.


* The key part Selection is to increase recall
* The key challange in most recall or top K selection is the large amout of candidate, some over hundreds of millions. so the model should be simple to calculate all candidates and speed is the most important parameter.

### Mutiple recall 

The current industrial usage is normally multiple recall system as shown below, so there will be multiple feature sets running and independently pick up top K. The K number is a super parameters that may need to be A/B tested

![multirecall](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/multirecall.jpg)

## Classical Recall Algorithms

1. LR

* Simple and useful. Easy to capture and represent features, easy to add business logic into feature sets.

* drawback: can not capture feature combination

![LR_recall](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LR_recall.jpg)

LR is most widely used in recall and CTR, so it is linear feature sets with manually introduced none linear combination

2. Improved LR to boost the generalization

* Mannually pick up the feature combination is complicated and time consuming. so put the feature combination into model

![LR_improve](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/LR_improve.jpg)

But the generalization will be a problem, if two feature combination has not appeared in training set, this model can not capture in serving stage. Especially for sparse feature set like in recall and CTR

3. FM(Factorization Machine) model

![FMmodel](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/FMmodel.jpg)

In Sparse embedding, although the two features may not appear in training, but as long as each feature is just a vector, it is still possible to capture the relation using dot product.

> This is essentailly key point for most embedding, to use dot product to measure similarity 

4. MF(Matrix Factorization)

Basically use two embedding, one for user and one for item/keywords

![MF](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/MF.jpg)

5. MF to FM

![MF2FM](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/MF2FM.jpg)

## Re-Ranking

Based on top K selected, use ranking algorithm mentioned above to rank, noticing selection algorithm will score for single documentation. pairwise or list wise will only be applied for re ranking

* The key part Selection is to increase precision

## balance between recall/precision

Some  search application focus on more precision. like "Trump", "Steve Jobs", in this case, selection part could be simplified, focusing on re ranking

Some  search application focus on more recall, like certain legal documentation, in this case we should focus on more recall

## High frequency vs Long tail

High frequency usually have high traffic volume, we can just use CTR or conversion rate to rank instead of modeling

# Deep Learning Based Ranking

Use deep learning to learning the hidden semantic representation for query/documentation   

## Emebedding representation

## CNN based semantic model

> A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval

* Traditional: Use BOW to abstract the feature from query or documentations

* Use Sliding window in CNN filter to abstract character level feature, not word level. So the model will convert word into a character level  embedding representation vector, then apply max pooling

![CNNsemantic](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/CNNsemantic.png)

## Learning local representation

# Evaluation

## Offline

### ROC

* Ranking precision: Out of all Documentation judged as relevant, how much of them are marked as relevant.
* Ranking Recall: Out of all relevant Documentation, how much of them have been marked as relevant

> Top K

In reality, K will be like 3,5,10, but the total documentation size could be very large. So how to choose the K documentation is very important

### F1

F-score or F-measure, is The weighted harmonic mean of precision and recall
![F1 score](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/F1score.svg)

### NDCG(Normalized Discounted Cumulative Gain)

Compared with only Binary classification, we can use multi-class labels, like 5 levels

Two assumptions are made in using DCG and its related measures.
* Highly relevant documents are more useful when appearing earlier in a search engine result list (have higher ranks)
* Highly relevant documents are more useful than marginally relevant documents, which are in turn more useful than non-relevant documents.

So in NDCG, the highly relevant is valued more than un relevant. So highly relevant documentation needs to be ranked in highest position, and most un-relevant documentation needs to be bottom. every wrong ranking will be punished.

## Online

# CTR(Click through rate)

* Widely used in Ads, Recommendation, Feed

## Unique in CTR

* Very Sparse discrete Features
* High order sparse Features
* Feature engineering is very important(conjecture combination features )

> example: Feature vector in CTR

![featurevector](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/featurevector.png)


## Classic Method

### Logistic regression

* Simple and useful. Easy to capture and represent features, easy to add business logic into feature sets.

* drawback: can not capture feature combination

> we can add the feature combination as pair of feature combination

* But then it dose not have good generalization capability, let's see if we did not see any __x_i__ and __x_j__ combination feature in training, then weight __w_i,j__ will be 0, and if we had this feature in serving, LR can not recognize it.

### Factor Machine

![FM](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/FM.png)


### GBDT

## Deep Learning on CTR

### Deep Learning feature set

* continuous: age, height, etc,

### Handle the challenge

* High order sparse feature space -> One hot encoding to Dense(embedding)

* Feature Engineering -> Deep NN automatically abstract features

* Feature Engineering: How to abstract low order feature combination -> FM in DNN

* Feature Engineering: How to abstract high order feature combination -> Deep NN



* discrete: professional, gender, college

### one hot to dense

One hot is good at representing discrete features. however, if we directly use one hot encoding in deep learning, the parameter size will be too large.

So the proper solution is 
* to convert ___one hot->Dense___
* different feature categories do not fully connected with each other 

![onehot2dense](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/onehot2dense.png)

### General DNN network Architecture

* one hot to dense
* add continous feature vector

> This is the general DNN architecture, it is also seen as Factorisation-Machine Supported Neural Networks

![DNN_CTR](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/DNN_CTR.png)

### Advanced architecture based on general architecture 

> The idea is to abstract the low order feature combination separately and add into general DNN structure 

#### Parallel 

![para_DNN_CTR](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/para_DNN_CTR.png)


#### Serialization 


## DNN model training and optimization 




 





