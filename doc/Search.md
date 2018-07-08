


> Query-Document Understanding

> Ranking

> Evaluation


# Query-Document Understanding

Mostly unsupervised way. To understand the Query and Documentation

## TF-IDF

## BM25

# Learn to Ranking

Learning to rank or machine-learned ranking (MLR) is the application of machine learning, typically supervised, semi-supervised or reinforcement learning, in the construction of ranking models for information retrieval systems.Training data consists of lists of items with some partial order specified between items in each list. This order is typically induced by giving a numerical or ordinal score or a binary judgment (e.g. "relevant" or "not relevant") for each item

Compared with unsupervised learning like TF-IDF and BM25. It will leverage the supervised learning scheme.


## Pointwise Learning to Rank

Turn the ranking to Binary Classification problem. The Training model will be pair of {keywords, Documentation}. The results is binary  either relevant or not.

> Regarding the Evaluation, we can evaluate the results from precision and recall.

* Ranking precision: Out of all Documentation judged as relevant, how much of them are marked as relevant.
* Ranking Recall: Out of all relevant Documentation, how much of them have been marked as relevant

> Different than other learning to rank problem, for each keywords, will only evaluate Top K Documentation.  

In reality, K will be like 3,5,10, but the total documentation size could be very large. So how to choose the K documentation is very important

## PairWise Learning to Rank

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

### nDCG(Normalized Discounted Cumulative Gain)

## Online

### CTR(Click through rate)
