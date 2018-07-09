


> Query-Document Understanding

> Ranking

> Evaluation

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

## Documentation understanding

# Learn to Ranking

Learning to rank or machine-learned ranking (MLR) is the application of machine learning, typically supervised, semi-supervised or reinforcement learning, in the construction of ranking models for information retrieval systems.Training data consists of lists of items with some partial order specified between items in each list. This order is typically induced by giving a numerical or ordinal score or a binary judgment (e.g. "relevant" or "not relevant") for each item

Compared with unsupervised learning like TF-IDF and BM25. It will leverage the supervised learning scheme.


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

## LambadaRank/LambadaMART

The idea is to learn the diff where two documentations have been ranked in wrong order. In this process, we do not need to know the real target function, only the gradient is enough

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

### CTR(Click through rate)
