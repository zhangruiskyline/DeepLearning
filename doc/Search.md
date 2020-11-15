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








 





