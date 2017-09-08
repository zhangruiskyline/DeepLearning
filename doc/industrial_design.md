<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Summary](#summary)
  - [Architecture](#architecture)
  - [Data](#data)
  - [Features](#features)
  - [models](#models)
  - [Evaluation](#evaluation)
  - [Discussion](#discussion)
- [Design Twitter](#design-twitter)
  - [Overall solution picture](#overall-solution-picture)
  - [FanOut: Push and Pull](#fanout-push-and-pull)
  - [Follow-up on details](#follow-up-on-details)
- [Twitter Feed Rank](#twitter-feed-rank)
  - [Requirements/Consideration :](#requirementsconsideration-)
  - [Model](#model)
  - [features](#features)
  - [Deep Learning](#deep-learning)
    - [Challenge: Sparse data](#challenge-sparse-data)
    - [Solve with deep learning](#solve-with-deep-learning)
  - [Quality Measurement](#quality-measurement)
  - [Platform](#platform)
    - [System Consideration](#system-consideration)
    - [Bottleneck](#bottleneck)
  - [Deployment and iterate](#deployment-and-iterate)
  - [Speed vs quality](#speed-vs-quality)
- [Design a iphone APP recommendation system(Siri Recommendation)](#design-a-iphone-app-recommendation-systemsiri-recommendation)
- [Facebook Feed Rank System](#facebook-feed-rank-system)
  - [Overall consideration](#overall-consideration)
  - [Data Model](#data-model)
  - [Feeds Ranking](#feeds-ranking)
    - [Edge rank system](#edge-rank-system)
  - [Production](#production)
    - [Push and Pull](#push-and-pull)
    - [Select FanOut](#select-fanout)
    - [Scalability](#scalability)
- [Spotify Music Recommendation](#spotify-music-recommendation)
  - [collaborative filter](#collaborative-filter)
  - [Content based](#content-based)
  - [](#)
- [YouTube Recommendation](#youtube-recommendation)
  - [Challenges](#challenges)
  - [Overview](#overview)
  - [Candidate generations](#candidate-generations)
    - [Model overview](#model-overview)
    - [Model Architecture](#model-architecture)
    - [Features](#features-1)
    - [NN architecture](#nn-architecture)
    - [Model Summary](#model-summary)
  - [Ranking](#ranking)
    - [Feature Engineering](#feature-engineering)
    - [Embedding Categorical Features](#embedding-categorical-features)
    - [Normalizing Continuous Features](#normalizing-continuous-features)
- [Quora Machine Learning Applications](#quora-machine-learning-applications)
  - [Semantic Question Matching with Deep Learning](#semantic-question-matching-with-deep-learning)
  - [How Quora build recommendation system](#how-quora-build-recommendation-system)
    - [Goal and data model](#goal-and-data-model)
    - [Feed Ranking](#feed-ranking)
    - [Ranking algorithm](#ranking-algorithm)
    - [feature](#feature)
  - [Answer Ranking](#answer-ranking)
  - [Ask to Answer](#ask-to-answer)
- [Pinteret: Smart Feed](#pinteret-smart-feed)
- [Amazon Deep Learning For recommendation](#amazon-deep-learning-for-recommendation)
  - [Overview](#overview-1)
  - [Architecture](#architecture-1)
  - [Deep learning with DSSTNE](#deep-learning-with-dsstne)
  - [Model parallel training Example](#model-parallel-training-example)
- [MeiTuan: Deep Learning on recommendation](#meituan-deep-learning-on-recommendation)
  - [Requirements and Scenario](#requirements-and-scenario)
  - [System Architecture](#system-architecture)
    - [Recall layer(Candidates Selections via collaborative filter)](#recall-layercandidates-selections-via-collaborative-filter)
    - [Ranking layer](#ranking-layer)
  - [Deep Learning on ranking](#deep-learning-on-ranking)
    - [Current system limitation](#current-system-limitation)
    - [How to generate label data](#how-to-generate-label-data)
    - [Feature selection](#feature-selection)
      - [Feature extraction](#feature-extraction)
      - [Feature combination](#feature-combination)
    - [Optimization and Loss function](#optimization-and-loss-function)
    - [Deep Learning system](#deep-learning-system)
- [Uber: Machine Learning system Michelangelo](#uber-machine-learning-system-michelangelo)
  - [System architecture](#system-architecture)
  - [WorkFlow](#workflow)
  - [Manage data](#manage-data)
    - [Offline](#offline)
    - [online](#online)
    - [Shared feature store](#shared-feature-store)
    - [Domain specific language for feature selection and transformation](#domain-specific-language-for-feature-selection-and-transformation)
  - [Train Model](#train-model)
  - [Evaluate Model](#evaluate-model)
  - [Deploy Model](#deploy-model)
  - [Prediction and Serving](#prediction-and-serving)
  - [Scale and Latency](#scale-and-latency)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Summary

How to approach the machine learning design problem

## Architecture

Big picture of whole system

1. data
2. model
3. feature
4. evaluation

## Data

how to get data, how to process, augmentation, etc.  if there is no label, how to get implicit data label

## Features

What are features, feature dimension, input and output of feature. How to process data into feature(clean, normalization,etc.)

Raw data feature or high level abstraction

## models

which model to choose, Tree based or deep learning, if it is deep learning, what could be the depth of network, what is NN architecture

## Evaluation

How to evaluate the model, what are main focus: accurate? speed?

## Discussion

1. How to make the mode fast?

# Design Twitter

http://blog.gainlo.co/index.php/2016/02/17/system-design-interview-question-how-to-design-twitter-part-1/

http://blog.gainlo.co/index.php/2016/02/24/system-design-interview-question-how-to-design-twitter-part-2/

The common strategy I would use here is to divide the whole system into several core components. There are quite a lot divide strategies, for example, you can divide by frontend/backend, offline/online logic etc.

## Overall solution picture

In this question, I would design solutions for the following two things:

1. Data modeling.

Data modeling – If we want to use a relational database like MySQL, we can define user object and feed object. Two relations are also necessary. One is user can follow each other, the other is each feed has a user owner.

2. How to serve feeds.

Serve feeds – The most straightforward way is to fetch feeds from all the people you follow and render them by time.

## FanOut: Push and Pull

* Push:  Push its tweet to all followers, O(n) write, O(1) read, Cons: if a user has many followers
* Pull: only post tweet to my own timeline, each user to pull tweets from all its followings
O(1) write. O(n) read. Cons: if a user follows many people

## Follow-up on details

1. When users followed a lot of people, fetching and rendering all their feeds can be costly. How to improve this?

There are many approaches. Since Twitter has the infinite scroll feature especially on mobile, each time we only need to fetch the most recent N feeds instead of all of them. Then there will many details about how the pagination should be implemented.

2. How to detect fake users?

This can be related to machine learning. One way to do it is to identify several related features like registration date, the number of followers, the number of feeds etc. and build a machine learning system to detect if a user is fake.

3. Feed Rank

Please refer [Design a Machine learning system](#design-a-machine-learning-system)

* How to measure the algorithm? Maybe by the average time users spend on Twitter or users interaction like favorite/retweet.
* What signals to use to evaluate how likely the user will like the feed? Users relation with the author, the number of replies/retweets of this feed, the number of followers of the author etc. might be important.
* If machine learning is used, how to design the whole system?

4. Trend Topics

first, How to get trending topic candidates? second, How to rank those candidates?

5. Who to follow

This is a core feature that plays an important role in user onboarding and engagement.

There are mainly two kinds of people that Twitter will show you – people you may know (friends) and famous account (celebrities/brands…).

The question would be how to rank them given that each time we can only show a few suggestions. I would lean toward using a machine learning system to do that.

There are tons of features we can use, e.g. whether the other person has followed this user, the number of common follows/followers, any overlap in basic information (like location) and so on so forth.

This is a complicated problem and there are various follow-up questions:

* How to scale the system when there are millions/billions of users?
* How to evaluate the system?
* How to design the same feature for Facebook (bi-directional relationship)

6. Search

The high-level approach can be pretty similar to Google search except that you don’t need to crawl the web. Basically, you need to build indexing, ranking and retrieval.

The most straightforward approach is to give each feature/signal a weight and then compute a ranking score for each tweet. Then we can just rank them by the score. Features can include reply/retweet/favorite numbers, relevance, freshness, users popularity etc..

But how do we evaluate the ranking and search? I think it’s better to define few core metrics like total number of searches per day, tweet click even followed by a search etc. and observe these metrics every day. They are also stats we should care whatever changes are made.

# Twitter Feed Rank

Based on Twitter Machine Learning tweets rank system design experience:

https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines.html?utm_campaign=Revue%2520newsletter&utm_medium=Newsletter&utm_source=revue

## Requirements/Consideration :

prediction models have to meet many requirements before they can be run in production at Twitter’s scale. This includes:
* Quality and speed of predictions
* Resource utilization
* Maintainability

Besides the prediction models themselves, a similar set of requirements is applied to the machine learning frameworks – that is, a set of tools that allows one to define, train, evaluate, and launch a prediction model. Specifically, we pay attention to:
* Training speed and scalability to very large datasets
* Extensibility to new techniques
* Tooling for easy training, debugging, evaluation and deployment


## Model

With ranking, we add an extra twist. Right after gathering all Tweets, each is scored by a relevance model. The model’s score predicts how interesting and engaging a Tweet would be specifically to you. A set of highest-scoring Tweets is then shown at the top of your timeline, with the remainder shown directly below.

## features
* The Tweet itself: its recency, presence of media cards (image or video), total interactions (e.g. number of Retweets or likes)
* The Tweet’s author: your past interactions with this author, the strength of your connection to them, the origin of your relationship
* You: Tweets you found engaging in the past, how often and how heavily you use Twitter

## Deep Learning

Deep learning modules can be composed in various ways (stacked, concatenated, etc.) to form a computational graph. The parameters of this graph can then be learned, typically by using back-propagation and SGD (Stochastic Gradient Descent) on mini-batches.

### Challenge: Sparse data

Tweet ranking lives in a different domain than what most researchers and deep learning algorithm usually focus on. This is mostly because the data is inherently sparse. For multiple reasons including availability and latency requirements, the presence of a feature cannot be guaranteed for each data record going through the model.

Usually, these problems are solved using other types of algorithm such as decision trees, logistic regression, feature crossing and discretization.

### Solve with deep learning

* Discretization:

sparse feature values can be wildly different from one data record to another. We found a way to discretize the input’s sparse features, before feeding them to the main deep net.

* A custom sparse linear layer:

this layer has two main extras compared to other sparse layers out there, namely an online normalization scheme that prevents gradients from exploding, and a per-feature bias to distinguish between the absence of a feature and the presence of a zero-valued feature.

* A sampling scheme associated with a calibration layer:

deep nets usually explore the space of solutions much better when the training dataset contains a similar number of positive and negative examples. However, hand-tuning a training dataset in this manner leads to uncalibrated output predictions. Thus, we added a custom isotonic calibration layer to recalibrate and output actual probabilities.
* A training plan:

with all these additions, the whole training procedure of a model now has multiple steps: discretizer calibration, deep net training, isotonic calibration of the predictions, and testing. Thanks to the flexibility of our platform, it is easy to declare these steps and run them in sequence.

## Quality Measurement


* Accurate Metrics

First, we evaluate the model using a well-defined accuracy metric we compute during model training. This measure tells us how well the model performs its task – specifically, giving engaging Tweets a high score. While final model accuracy is a good early indicator, it cannot be used alone to reliably predict how people using Twitter will react to seeing Tweets picked out by that model

Impact on people using Twitter is typically measured by running one or more A/B tests and comparing the results between experiment buckets.

Finally, even if the desired real-time speed could be achieved for a given model quality, launching that model would be subject to the same trade-off analysis as any new feature. We would want to know the impact of the model and weigh that against any increase in the cost of running the model. The added cost can come from higher hardware utilization, or more complicated operation and support.

## Platform

### System Consideration

* Distributed training

### Bottleneck

Analyzed Bottleneck: CPU bound service or IO bound service

## Deployment and iterate

* Before roll out

Models have to meet many requirements before they can be run in production

  * Quality and speed of predictions
  * Resource utilization
  * Maintainability

* A/B test for user experience
* Measure impact:

e.g: Impact on people using Twitter is typically measured by running one or more A/B tests and comparing the results between experiment buckets.

* Offline vs online

log data, offline training for accuracy model. online serving

## Speed vs quality

* Lots of tasks need latency requirements. quick serving time may be more important than accuracy

* Also Training time could be important, as we may have urgent requirement for some tasks like fraud detection, need to wrap up model quickly and push online, rather than design best detection model

* Twitter Example

Even if the desired real-time speed could be achieved for a given model quality, launching that model would be subject to the same trade-off analysis as any new feature. We would want to know the impact of the model and weigh that against any increase in the cost of running the model. The added cost can come from higher hardware utilization, or more complicated operation and support.



# Design a iphone APP recommendation system(Siri Recommendation)

1. overall system:

* Training data: each user click data/select or not . and their feature list,
* Model:  NN model. GBT
* Output:  regression/classification,

offline model. for training,  online mode, for serving

how to measure accuracy

2. Feature selection

* static features: ID, gender, age, country,
* dynamic: time, location,

3. Optimization:

* data

100M iphone, each record is 10 Bytes, 1000 records per day/user, 1T per day/user.
how to compress: only upload once per day, do local data analysis/aggregation first

* networking:

if model is heavy, then probably need a local model, with rough estimation, like local model with less accuracy.

* Real time

need to be fast, could start after take out phone instead of unlock


# Facebook Feed Rank System

http://blog.gainlo.co/index.php/2016/03/29/design-news-feed-system-part-1-system-design-interview-questions/

http://blog.gainlo.co/index.php/2016/04/05/design-news-feed-system-part-2/

## Overall consideration

it’s better to have some high-level ideas by dividing the big problem into subproblem.

* Data model. We need some schema to store user and feed object. More importantly, there are lots of trade-offs when we try to optimize the system on read/write. I’ll explain in details next.
* Feed ranking. Facebook is doing more than ranking chronologically.
* Feed publishing. Publishing can be trivial when there’re only few hundreds of users. But it can be costly when there are millions or even billions of users. So there’s a scale problem here.

## Data Model

* Objects: User and Feeds
  * User object, we can store userID, name, registration date and so on so forth.
  * Feed object, there are feedId, feedType, content, metadata etc., which should support images and videos as well.

* Data structure to store data
  * user-feed relation. We can create a user-feed table that stores userID and corresponding feedID. For a single user, it can contain multiple entries if he has published many feeds.

  * Friends relation: adjacency list is one of the most common approaches.We can use a friend table that contains two userIDs in each entry to model the edge (friend relation). By doing this, most operations are quite convenient like fetch all friends of a user, check if two people are friends.

![adjacent_list](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/adjacent_list.png)

* Data fetch

  * 3 steps classical approach:

  The system will first get all userIDs of friends from friend table. Then it fetches all feedIDs for each friend from user-feed table. Finally, feed content is fetched based on feedID from feed table.

  * denormalization optimization

  store feed content together with feedID in user-feed table so that we don’t need to join the feed table any more.

  cons:

  1. Data redundancy. We are storing redundant data, which occupies storage space (classic time-space trade-off).
  2. Data consistency. Whenever we update a feed, we need to update both feed table and user-feed table. Otherwise, there is data inconsistency. This increases the complexity of the system.



## Feeds Ranking

* Intention:

why do we want to change the ranking? How do we evaluate whether the new ranking algorithm is better? It’s definitely impressive if candidates come up with these questions by themselves.

Let’s say there are several core metrics we care about, e.g. users stickiness, retention, ads revenue etc.. A better ranking system can significantly improve these metrics potentially, which also answers how to evaluate if we are making progress.

* Approach

A common strategy is to calculate a feed score based on various features and rank feeds by its score, which is one of the most common approaches for all ranking problems.

* Features

More specifically, we can select several features that are mostly relevant to the importance of the feed, e.g. share/like/comments numbers, time of the update, whether the feed has images/videos etc.. And then, a score can be computed by these features, maybe a linear combination. This is usually enough for a naive ranking system.

### Edge rank system

As you can see that what matters here are two things – features and calculation algorithm. To give you a better idea of it, I’d like to briefly introduce how ranking actually works at Facebook-Edge Rank.

For each news update you have, whenever another user interacts with that feed, they’re creating what Facebook calls an Edge, which includes actions like like and comments.


Edge Rank basically is using three signals: affinity score, edge weight and time decay.

* Affinity score (u).

For each news feed, affinity score evaluates how close you are with this user. For instance, you are more likely to care about feed from your close friends instead of someone you just met once. You might ask how affinity score is calculated, I’ll talk about it soon.

  * First of all, explicit interactions like comment, like, tag, share, click etc. are strong signals we should use. Apparently, each type of interaction should have different weight. For instance, comments should be worth much more than likes.

  * Secondly, we should also track the time factor. Perhaps you used to interact with a friend quite a lot, but less frequent recently. In this case, we should lower the affinity score. So for each interaction, we should also put the time decay factor.

* Edge weight (e).

Edge weight basically reflects importance of each edge. For instance, comments are worth more than likes.

* Time decay (d).

The older the story, the less likely users find it interesting.

![edge_rank](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/edge_rank.png)

> Some reference

[Blog in Chinese to explain](https://zhuanlan.zhihu.com/p/20901694)



## Production

### Push and Pull

* Push

For a push system, once a user has published a feed, we immediately pushing this feed (actually the pointer to the feed) to all his friends. The advantage is that when fetching feed, you don’t need to go through your friends list and get feeds for each of them. It __significantly reduces read operation__. However, the downside is also obvious. It __increases write operation__ especially for people with a large number of friends.

* Pull

For a pull system, feeds are only fetched when users are loading their home pages. So feed data doesn’t need to be sent right after it’s created. You can see that this approach __optimizes for write operation__, but can be quite slow to fetch data even after using denormalization

### Select FanOut

The process of pushing an activity to all your friends or followers is called a fanout. So the push approach is also called fanout on write, while the pull approach is fanout on load.

if you are mainly using push model, what you can do is to disable fanout for high profile users and other people can only load their updates during read.

The idea is that push operation can be extremely costly for high profile users since they have a lot of friends to notify. By disabling fanout for them, we can save a huge number of resources. Actually Twitter has seen great improvement after adopting this approach.

By the same token, once a user publish a feed, we can also limit the fanout to only his active friends. For non-active users, most of the time the push operation is a waste since they will never come back consuming feeds

### Scalability

You want to minimize the number of disk seeks that need to happen when loading your home page. The number of seeks could be 0 or 1 but definitely not O(num friends).

https://www.quora.com/What-are-the-scaling-issues-to-keep-in-mind-while-developing-a-social-network-feed

# Spotify Music Recommendation

 [Spotify Music recommendation](http://benanne.github.io/2014/08/05/spotify-cnns.html)

## collaborative filter

Traditionally, Spotify has relied mostly on collaborative filtering approaches to power their recommendations. The idea of collaborative filtering is to determine the users’ preferences from historical usage data.

Pure collaborative filtering approaches do not use any kind of information about the items that are being recommended, except for the consumption patterns associated with them: they are content-agnostic.

Unfortunately, this also turns out to be their biggest flaw. Because of their reliance on usage data, popular items will be much easier to recommend than unpopular items, as there is more usage data available for them. This is usually the opposite of what we want.

## Content based

There is quite a large semantic gap between music audio on the one hand, and the various aspects of music that affect listener preferences on the other hand. Some of these are fairly easy to extract from audio signals, such as the genre of the music and the instruments used. Others are a little more challenging

##

# YouTube Recommendation

[Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html)

## Challenges
* Scale:

Many existing recommendation algorithms proven to work well on small problems fail to operate on our scale. Highly specialized distributed learning algorithms and efficient serving systems are essential for handling YouTube’s massive user base and corpus.

* Freshness

YouTube has a very dynamic corpus with many hours of video are uploaded per second. The recommendation system should be responsive enough to model newly uploaded content as well as the latest actions taken by the user.

* Noise

Historical user behavior on YouTube is inher- ently difficult to predict due to sparsity and a vari- ety of unobservable external factors. We rarely ob- tain the ground truth of user satisfaction and instead model noisy implicit feedback signals. Furthermore, metadata associated with content is poorly structured without a well defined ontology. Our algorithms need to be robust to these particular characteristics of our training data.

## Overview
![youtube_recommend](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/youtube_recommend.png)

The system is comprised of two neural networks: one for candidate generation and one for ranking.

The candidate generation network takes events from the user’s YouTube activity history as input and retrieves a small subset (hundreds) of videos from a large corpus. These candidates are intended to be generally relevant to the user with high precision. The candidate generation network only provides broad personalization via collaborative filtering. The similarity between users is expressed in terms of coarse features such as IDs of video watches, search query tokens and demographics.

Presenting a few “best” recommendations in a list requires a fine-level representation to distinguish relative importance among candidates with high recall. The ranking network accomplishes this task by assigning a score to each video according to a desired objective function using a rich set of features describing the video and user. The highest scoring videos are presented to the user, ranked by their score.

* How to measure
  * During development, we make extensive use of offline metrics (precision, recall, ranking loss, etc.) to guide iterative improvements to our system.
  * However for the final determination of the effectiveness of an algorithm or model, we rely on A/B testing via live experiments. we can measure subtle changes in click-through rate, watch time, and many other metrics that measure user engagement.
  * Live A/B test results may not always correlated with offline experiments



## Candidate generations

During candidate generation, the enormous YouTube corpus is winnowed down to hundreds of videos that may be relevant to the user. It could be a was a __matrix factorization__ approach trained under rank loss

### Model overview

* pose recommendation as extreme multiclass classification where the prediction problem


![recommendation_classfication](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/recommendation_classfication.png)

where u represents a high-dimensional “embedding”of the user, context pair and the vj represent embeddings of each candidate video.

In this setting, an embedding is simply a mapping of sparse entities (individual videos, users etc.) into a dense vector. The task of the deep neural network is to learn user embeddings u as a function of the user’s history and context that are useful for discriminating among videos with a softmax classifier.

Although explicit feedback mechanisms exist on YouTube (thumbs up/down, in-product surveys, etc.) we use the implicit feedback of watches to train the model, where a user completing a video is a positive example.

To efficiently train such a model with __millions of classes__, we rely on a technique to sample negative classes from the background distribution (“candidate sampling”)

* Serving stage

At serving time we need to compute the most likely N classes (videos) in order to choose the top N to present to the user. Scoring millions of items under a strict serving latency of tens of milliseconds requires an approximate scoring scheme sublinear in the number of classes.

* the scoring problem reduces to a nearest neighbor search in the dot product space

### Model Architecture

* user embedding

we learn high dimensional embeddings for each video in a fixed vocabulary and feed these embeddings into a feedforward neural network. A user’s watch history is represented by a variable-length sequence of sparse video IDs which is mapped to a dense vector representation via the embeddings. The network requires fixed-sized dense inputs and simply averaging the embeddings performed best among several strategies.

### Features

A key advantage of using deep neural networks as a generalization of matrix factorization is that arbitrary continuous and categorical features can be easily added to the model.

* General features

Search history is treated similarly to watch history, Each query is tokenized into unigrams and bigrams and each to- ken is embedded. Once averaged, the user’s tokenized, embedded queries represent a summarized dense search history.

* users Features

Demographic features are important for providing priors so that the recommendations behave reasonably for new users. The user’s geographic region and device are embedded and concatenated. Simple binary and continuous features such as the user’s gender, logged-in state and age are input di- rectly into the network as real values normalized to [0, 1].

* video features/New Video

We consistently observe that users prefer fresh content, though not at the expense of relevance.

Machine learning systems often exhibit an implicit bias towards the past because they are trained to predict future behavior from historical examples. To correct for this, we feed the __age of the training example__ as a feature during training.

At serving time, this feature is set to zero (or slightly negative) to reflect that the model is making predictions at the very end of the training window.

### NN architecture
Training examples are generated from all YouTube watches (even those embedded on other sites) rather than just watches on the recommendations we produce. Otherwise, it would be very difficult for new content to surface and the recommender would be overly biased towards exploitation.

Another key insight that improved live metrics was to generate a fixed number of training examples per user, effectively weighting our users equally in the loss function. (Another way to normalization)

* Explore NN structure and feature space

a vocabulary of 1M videos and 1M search tokens were embedded with 256 floats each in a maximum bag size of 50 recent watches and 50 recent searches.

The softmax layer outputs a multinomial distribution over the same 1M video classes with a dimension of 256 (which can be thought of as a separate output video embedding). These models were trained until convergence over all YouTube users, corresponding to several epochs over the data.

  * Depth 0: A linear layer simply transforms the concate- nation layer to match the softmax dimension of 256
  * Depth 1: 256 ReLU
  * Depth 2: 512 ReLU → 256 ReLU
  * Depth 3: 1024 ReLU → 512 ReLU → 256 ReLU
  * Depth 4: 2048 ReLU → 1024 ReLU → 512 ReLU → 256 ReLU



### Model Summary


![NN_recommendation](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/NN_recommendation.png)

Deep candidate generation model architecture showing embedded sparse features concatenated with dense features. Embeddings are averaged before concatenation to transform variable sized bags of sparse IDs into fixed-width vectors suitable for input to the hidden layers. All hidden layers are fully connected. In training, a cross-entropy loss is minimized with gradient descent on the output of the sampled softmax. At serving, an approximate nearest neighbor lookup is performed to generate hundreds of candidate video recommendations.

## Ranking

During ranking, we have access to many more features describing the video and the user’s relationship to the video because only a few hundred videos are being scored rather than the millions scored in candidate generation.

We use a deep neural network with similar architecture as candidate generation to assign an independent score to each video impression using logistic regression

![ranking_recommendation](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ranking_recommendation.png)

### Feature Engineering
use hundreds of features in our ranking mod- els, roughly split evenly between categorical and continuous.

We observe that the most important signals are those that describe a user’s previous interaction with the item itself and other similar items, matching others’ experience in ranking ads.

* Example:

As an example, consider the user’s past history with the channel that uploaded the video being scored - how many videos has the user watched from this channel? When was the last time the user watched a video on this topic? These continuous features describing past user actions on related items are particularly powerful because they generalize well across disparate items.

### Embedding Categorical Features

Similar to candidate generation, we use embeddings to map sparse categorical features to dense representations suitable for neural networks.

### Normalizing Continuous Features
A continuous feature x with distribution f is transformed to x by scaling the values such that the feature is equally distributed in [0,1) using the cumulative distribution

# Quora Machine Learning Applications

## Semantic Question Matching with Deep Learning

https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning

* Problem definition: To detect similar # QUESTION: More formally, the duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function:


```Math
f(q1, q2) → 0 or 1
```

* Current model

Our current production model for solving this problem is a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words.

Tips: questions may be different length, so we need to force them to same size.

* Improvement

We trained our own word embeddings using Quora's text corpus using LSTM, combined them to generate question embeddings for the two questions, and then fed those question embeddings into a representation layer. We then concatenated the two vector representation outputs from the representation layers and fed the concatenated vector into a dense layer to produce the final classification result.

![quora embeddings](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_1.png)

*  Tai, Socher empirically-motivated handcrafted features

  * the distance, calculated as the sum of squares of the difference of the two vectors, and * the “angle”, calculated as an element-wise multiplication of the two vector representations (denoted ʘ).

A neural network using the distance and angle as the two input neurons was then stacked on top

![quora embeddings 2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_2.png)

* Attention Model

Finally, we tried an attention-based approach from Google Research [4] that combined neural network attention with token alignment, commonly used in machine translation. The most prominent advantage of this approach, relative to other attention-based approaches, was the small number of parameters. Similar to the previous two approaches, this model represents each token from the question with a word embedding. The process, shown in Figure 3, trained an attention model (soft alignment) for all pairs of words in the two questions, compared aligned phrases, and finally aggregated comparisons to get classification result.

![quora attention](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_attention.png)


## How Quora build recommendation system

https://www.slideshare.net/xamat/recsys-2016-tutorial-lessons-learned-from-building-reallife-recommender-systems

### Goal and data model
The core consideration for Quora's recommendation model is like:

![quora recsys](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_recsys.png)

And the core data flow model is like
![quora data](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_data.png)

### Feed Ranking
* Personalized Feed Ranking. Present most interesting stories for a user at a given time
  * Interesting = topical relevance + social relevance + timeliness
  * Stories = questions + answers

*  Quora uses learning-to-rank
  * Compared with time based ranking ,relevance based is much better

* Challenges:
  * potentially many candidate stories
  * real-time ranking
  * optimize for relevance

* The basic data for ranking: Impression logs
![quora impression log](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/impression_log.png)

### Ranking algorithm
![quora ranking algorithm](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ranking_algorithm.png)

* And Quora has defined a relevance score algorithm as
![quora relevance](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_relevance.png)

* In Summary

This is a weighted sum of actions to predict user's interet to a story. There are two ways to do so:

1. predict final results
2. predict each actions(upvote, read, share. etc) and weight sum again

> The second one is more resource consuming and Explanation vice better

### feature
* Major feature categories
  * user (e.g. age, country, recent activity)
  * story (e.g. popularity, trendiness, quality)
  * interactions between the two (e.g. topic or author affinity)


* Implicit is always better than explicit
  * More dense, available for all users
  * Better representations of user vs user reflections
  * Better correlated with A/B test

## Answer Ranking

https://engineering.quora.com/A-Machine-Learning-Approach-to-Ranking-Answers-on-Quora

* Ground truth data: upvotes/downvotes

some problem:
1. time sensitive
2. rich get richer
3. joke answer
4. good answer from not so active user

* Goal:

In ranking problems our goal is to predict a list of documents ranked by relevance. In most cases there is also additional context, e.g. an associated user that's viewing results, therefore introducing personalization to the problem.

* System architecture
![quora_ranking](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_ranking.png)

* Ground Truth Build

1. A/B Test online observation
2. offline: user survey. manually create ranking, starting from some already known good one

* Good answer standard

1. Answers the question that was asked.
2. Provides knowledge that is reusable by anyone interested in the question.
3. Answers that are supported with rationale.
4. Demonstrates credibility and is factually correct.
5. Is clear and easy to read.

* Features

  * text based features:
  * expertise-based features
  * upvote/downvote history

General rule is make sure our features generalize well. Text features can be particularly problematic in this regard.

Ensemble always works better

* Real production

  * rank the new answer on the question page as soon as possible to provide a smooth user experience

1. simple model with easy to calculate feature as approximate, once answer is added, recompute the more accurate score asynchronously

2. Question pages can contain hundreds of answers.

  1. cache answer scores so that the question page can load in a reasonable time
  2. cache all features
  3. All this data (answer scores and feature values) is stored in HBase, which is an open-source NoSQL datastore able to handle a large volume of data and writes.
  4.  cache score sometimes is not good when answer or feature changes: Consider a user who has tens of thousands of answers on Quora. If we depend on a feature like the number of answers added by an answer author, then every time this user adds an answer, we have to update the score of all of their answers at once.stopped updating feature values if that wouldn't impact the answer score at all.

## Ask to Answer
https://engineering.quora.com/Ask-To-Answer-as-a-Machine-Learning-Problem

* Frame the Problem
Given a question and a viewer rank all other users based on how 「well-suited」 they are.
well-suited」= likelihood of viewer sending a request + likelihood of the candidate adding a good answer

Furthermore, we can derive this as:
```
w1⋅had_request+w2⋅had_answer+w3⋅answer_goodness+⋯
```
* Features:
descriptors of the question, the viewer, and the candidate. some of the most important features are history related - features based on what the viewer or candidate has done in the past

* Labels:
the result of the suggestion as a number (e.g. 1 for answer, 0 for no answer).

![quora_A2A](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_A2A.png)


# Pinteret: Smart Feed

https://medium.com/@Pinterest_Engineering/building-a-smarter-home-feed-ad1918fdfbe3

https://mp.weixin.qq.com/s?__biz=MzA4OTk5OTQzMg==&mid=2449231037&idx=1&sn=c2fc8a7d2832ea109e2abe4b773ff1f5#rd

# Amazon Deep Learning For recommendation

https://aws.amazon.com/blogs/big-data/generating-recommendations-at-amazon-scale-with-apache-spark-and-amazon-dsstne/

## Overview

In Personalization at Amazon, we use neural networks to generate personalized product recommendations for our customers. Amazon’s product catalog is huge compared to the number of products that a customer has purchased, making our datasets extremely sparse. And with hundreds of millions of customers and products, our neural network models often have to be distributed across multiple GPUs to meet space and time constraints.

On the other hand, data for training and prediction tasks is processed and generated from Apache Spark on a CPU cluster. This presents a fundamental problem: data processing happens on CPU while training and prediction happen on GPU.

## Architecture

![aws_dl_arch](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/aws_dl_arch.png)

* CPU Job:

In this architecture, data analytics and processing (i.e., CPU jobs) are executed through vanilla Spark on Amazon EMR, where the job is broken up into tasks and runs on a Spark executor.

* GPU Job:

GPU job above refers to the training or prediction of neural networks.

The partitioning of the dataset for these jobs is done in Spark, but the execution of these jobs is delegated to ECS and is run inside Docker containers on the GPU slaves. Data transfer between the two clusters is done through Amazon S3.

On the GPU node, each task does the following:

1. Downloads its data partition from S3.
2. Executes the specified command.
3. Uploads the output of the command back to S3.

## Deep learning with DSSTNE

![aws_dl_arch_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/aws_dl_arch_2.png)

Support large, sparse layers

* Model Training:

In model parallel training, the model is distributed across N GPUs – the dataset (e.g., RDD) is replicated to all GPU nodes. Contrast this with data parallel training where each GPU only trains on a subset of the data, then shares the weights with each other using synchronization techniques such as a parameter server.

* prediction

After the model is trained, we generate predictions (e.g., recommendations) for each customer. This is an embarrassingly parallel task as each customer’s recommendations can be generated independently. Thus, we perform data parallel predictions, where each GPU handles the prediction of a batch of customers.


## Model parallel training Example

See Link for detail





# MeiTuan: Deep Learning on recommendation

https://tech.meituan.com/dl.html

## Requirements and Scenario

## System Architecture

### Recall layer(Candidates Selections via collaborative filter)

Different recall methodologies will generate different candidate pools, and then apply the ranking to all of them


* user based collaborative-filter

Find N similar users to current user, and score item based on those N users' score results. use Jaccard Similarity for similar users. R_x and R_y are user's score on items

![Jaccard Similarity](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/Jaccard.png)


* Model based collaborative-filter

use embedding to calculate user and item vector, and calculate inner product for user i to item j.

* Query based

Abstract the user intention by query, wifi status, Geo information. etc

* Location based


### Ranking layer

![meituan_rank](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/meituan_rank.png)


## Deep Learning on ranking

### Current system limitation


If the ranking is only based on history data, it will be limited. The current system looks like
![meituan_ml](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/meituan_ml.png)

The none linear model and GBDT can not beat LR in CTR, and LR model representation capability is not strong

* Example of recommendation system issue

![miss_recommend](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/miss_recommend.png)

The below pic shows that recommendation system recommends some items user clicked before, but may not be good enough in current context(like too far away), so we need to have more complex features rather than simple distance or manual crafted features. New [wide deep learning model](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) is used. Deep learning can learn the low level features and transform to high level features. so the team explores this auto-feature selection

### How to generate label data

positive sample when clicked, negative when not clicked, purchased item will have added weight. The portion for positive/negative samples will be around 10% in order to prevent overfitting

### Feature selection

* User

Like gender, price preference, location, item preference

* item

local stores(prices, orders, reviews), orders(price, deliver time, volume)

* Context

user current location, search query. etc

#### Feature extraction


![feature_extraction](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/feature_extraction.png)

The overhead for feature selection, extraction and adding will be more and more, so CTR prediction will be harder and harder for more features added. so intend to use deep learning to automatically select features.

#### Feature combination

Combine features and transform features

* normalization

Min-Max, CDF

* Aggregation

Use super linear (__X^2__) and sub linear(__sqr(x)__)

### Optimization and Loss function


### Deep Learning system

![meituan_dl](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/meituan_dl.png)

# Uber: Machine Learning system Michelangelo

https://eng.uber.com/michelangelo/

Michelangelo is designed to address these gaps by standardizing the workflows and tools across teams though an end-to-end system that enables users across the company to easily build and operate machine learning systems at scale. Our goal was not only to solve these immediate problems, but also create a system that would grow with the business

## System architecture


Michelangelo consists of a mix of open source systems and components built in-house. The primary open sourced components used are HDFS, Spark, Samza, Cassandra, MLLib, XGBoost, and TensorFlow.

## WorkFlow

![uber_ml](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/uber_ml.png)


We designed Michelangelo specifically to provide scalable, reliable, reproducible, easy-to-use, and automated tools to address the following six-step workflow:  

* Manage data
* Train models
* Evaluate models
* Deploy models
* Make predictions
* Monitor predictions

## Manage data

Finding good features is often the hardest part of machine learning and we have found that building and managing data pipelines is typically one of the most costly pieces of a complete machine learning solution.

A platform should provide standard tools for building data pipelines to generate feature and label data sets for training (and re-training) and feature-only data sets for predicting.

The data management components of Michelangelo are divided between online and offline pipelines

* Offline: feed batch model training and batch prediction jobs
* Online: feed online, low latency predictions (and in the near future, online learning systems).

### Offline

Uber’s transactional and log data flows into an HDFS data lake and is easily accessible via Spark and Hive SQL compute jobs. We provide containers and scheduling to run regular jobs to compute features which can be made private to a project or published to the Feature Store (see below) and shared across teams, while batch jobs run on a schedule or a trigger and are integrated with data quality monitoring tools to quickly detect regressions in the pipeline–either due to local or upstream code or data issues

### online

Models that are deployed online cannot access data stored in HDFS, and it is often difficult to compute some features in a performant manner directly from the online databases that back Uber’s production services (for instance, it is not possible to directly query the UberEATS order service to compute the average meal prep time for a restaurant over a specific period of time). Instead, we allow features needed for online models to be precomputed and stored in Cassandra where they can be read at low latency at prediction time.

* Batch precompute.

The first option for computing is to conduct bulk precomputing and loading historical features from HDFS into Cassandra on a regular basis. This is simple and efficient, and generally works well for historical features where it is acceptable for the features to only be updated every few hours or once a day.

For example: UberEATS uses this system for features like a ‘restaurant’s average meal preparation time over the last seven days.

* Near-real-time compute.

The second option is to publish relevant metrics to Kafka and then run Samza based streaming compute jobs to generate aggregate features at low latency. These features are then written directly to Cassandra for serving and logged back to HDFS for future training jobs.

For Example: UberEATS uses this near-realtime pipeline for features like a ‘restaurant’s average meal preparation time over the last one hour.

### Shared feature store

* It allows users to easily add features they have built into a shared feature store, requiring only a small amount of extra metadata (owner, description, SLA, etc.) on top of what would be required for a feature generated for private, project-specific usage.
* Once features are in the Feature Store, they are very easy to consume, both online and offline, by referencing a feature’s simple canonical name in the model configuration.

At the moment, we have approximately 10,000 features in Feature Store that are used to accelerate machine learning projects

### Domain specific language for feature selection and transformation

Often the features generated by data pipelines or sent from a client service are not in the proper format for the model

* Example:

In some cases, it may be more useful for the model to transform a timestamp into an hour-of-day or day-of-week to better capture seasonal patterns.

In other cases, feature values may need to be normalized (e.g., subtract the mean and divide by standard deviation).

## Train Model

support offline, large-scale distributed training of decision trees, linear and logistic models, unsupervised models (k-means), time series models, and deep neural networks.

A model configuration specifies the model type, hyper-parameters, data source reference, and feature DSL expressions, as well as compute resource requirements (the number of machines, how much memory, whether or not to use GPUs, etc.). It is used to configure the training job, which is run on a YARN or Mesos cluster.

## Evaluate Model

visualization

## Deploy Model

* Offline deployment: The model is deployed to an offline container and run in a Spark job to generate batch predictions either on demand or on a repeating schedule.

* Online deployment: The model is deployed to an online prediction service cluster (generally containing hundreds of machines behind a load balancer) where clients can send individual or batched prediction requests as network RPC calls.

* Library deployment: We intend to launch a model that is deployed to a serving container that is embedded as a library in another service and invoked via a Java API. (It is not shown in Figure 8, below, but works similarly to online deployment).


## Prediction and Serving


## Scale and Latency

* Scale: add more hosts to the prediction service cluster and let the load balancer spread the load. In the case of offline predictions, we can add more Spark executors and let Spark manage the parallelism.

* Latency: In the case of a model that does not need features from Cassandra, we typically see P95 latency of less than 5 milliseconds (ms). In the case of models that do require features from Cassandra, we typically see P95 latency of less than 10ms. The highest traffic models right now are serving more than 250,000 predictions per second.
