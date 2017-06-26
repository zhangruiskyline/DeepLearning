<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Model selection and data prepare](#model-selection-and-data-prepare)
  - [SoftMax and cross-entropy](#softmax-and-cross-entropy)
    - [SoftMax](#softmax)
    - [Cross-Entropy](#cross-entropy)
- [Bias vs Variance](#bias-vs-variance)
  - [Learn from learning curves](#learn-from-learning-curves)
  - [Debug learning algorithm](#debug-learning-algorithm)
    - [Diagnosing Neural Networks](#diagnosing-neural-networks)
    - [Model Complexity Effects:](#model-complexity-effects)
- [Overfitting](#overfitting)
  - [How to overcome overfitting](#how-to-overcome-overfitting)
- [Advantages of some particular algorithms](#advantages-of-some-particular-algorithms)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Model selection and data prepare

Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set.

* Training set: a set of examples used for learning: to fit the parameters of the classifier In the MLP case, we would use the training set to find the “optimal” weights with the back-prop rule

Optimize the parameters in Θ using the training set for each polynomial degree.

* Validation set: a set of examples used to tune the parameters of a classifier In the MLP case, we would use the validation set to find the “optimal” number of hidden units or determine a stopping point for the back-propagation algorithm.

Find the polynomial degree d with the least error using the cross validation set.

* Test set: a set of examples used only to assess the performance of a fully-trained classifier In the MLP case, we would use the test to estimate the error rate after we have chosen the final model (MLP size and actual weights) After assessing the final model on the test set, YOU MUST NOT tune the model any further!

Estimate the generalization error using the test set with Jtest(Θ(d)), (d = theta from polynomial with lower error);

Why separate test and validation sets? The error rate estimate of the final model on validation data will be biased (smaller than the true error rate) since the validation set is used to select the final model After assessing the final model on the test set, YOU MUST NOT tune the model any further!

## SoftMax and cross-entropy

### SoftMax

Softmax is widely used in last activation layer for Deep NN network, targets to accomplish multi-class classification problem

In mathematics, the softmax function, or normalized exponential function,[1]:198 is a generalization of the logistic function that "squashes" a K-dimensional vector  of arbitrary real values to a K-dimensional vector of real values in the range (0, 1] that add up to 1.

In probability theory, the output of the softmax function can be used to represent a categorical distribution – that is, a probability distribution over K different possible outcomes. The softmax function is used in various multiclass classification methods, such as multinomial logistic regression,[1]:206–209 multiclass linear discriminant analysis, naive Bayes classifiers, and artificial neural networks.

![softmax](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/softmax.png)

### Cross-Entropy

Once we have done the classification via softmax, we need to know how well it performs. cross-entropy is to see it as a (minus) log-likelihood for the data y′i, under a model yi, if we have K_n different classes, Then the cross entropy will be derived as

![cross entropy](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/cross_entropy.png)

* Cross Entropy for logistic regression binary classification

![LR cross entropy](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lr_cross_entropy.png)





# Bias vs Variance

* We need to distinguish whether bias or variance is the problem contributing to bad predictions.

* High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.

  * High bias: Both __training error__ and __cross_validation__ error will be __high__
  * high Variance : __training error__ is low and  __cross_validation__ is much higher than training error

* Choice of lambda: too large: underfitting, too small or 0, overfitting.

## Learn from learning curves
  * If learning algorithm suffer high bias. getting more data dose not help much

![high_bias](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/high_bias.png)

  * If learning algorithm suffer high variance, getting more data is likely to help

![high_variance](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/high_variance.png)

## Debug learning algorithm

* example of using LR in test: get low training error but large test error. So here is step to debug

1. Getting more training examples: Fixes high variance
2. Trying smaller sets of features: Fixes high variance
3. Adding features: Fixes high bias
4. Adding polynomial features: Fixes high bias
5. decreasing λ: Fixes high bias
6. Increasing λ: Fixes high variance.

### Diagnosing Neural Networks

1. A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
2. A large neural network with more parameters is prone to overfitting. It is also computationally expensive.
In this case you can use regularization (increase λ) to address the overfitting.

Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

### Model Complexity Effects:

1. Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
2. Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.

In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

# Overfitting

## How to overcome overfitting

* Keep it Simple

Go for simpler models over more complicated models. Generally, the fewer parameters that you have to tune the better. in theory, lower VC dimension will likely to overcome overfitting

* Cross-Validation


* Regularization/Penalizes the complexity

  * L1: Tends to train lots of parameters into 0, sparse model
  * L2: tends to train parameters to be small, so no "big" features
  * SVM: Squared norm of the weight vector in an SVM

* Ensemble model

* Dropout for NN

* Decision Tree tends to have overfitting: The ID3 algorithm will grow each branch in the tree just deep enough to classify the training instances perfectly
  * Pre_prune: stop growing the tree earlier, before it perfectly classifies the training set.
  * Post_prune: allows the tree to perfectly classify the training set, and then post prune the tree

# Advantages of some particular algorithms

* Advantages of Naive Bayes:

Super simple, you're just doing a bunch of counts. If the NB __*conditional independence assumption actually holds*__, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn't hold, a NB classifier still often performs surprisingly well in practice. A good bet if you want to do some kind of semi-supervised learning, or want something embarrassingly simple that performs pretty well.

* Advantages of Logistic Regression:

Lots of ways to regularize your model, and you don't have to worry as much about your features being correlated, like you do in Naive Bayes. You also have a nice probabilistic interpretation, unlike decision trees or SVMs, and you can easily update your model to take in new data (using an online gradient descent method), again unlike decision trees or SVMs. Use it if you want a probabilistic framework (e.g., to easily adjust classification thresholds, to say when you're unsure, or to get confidence intervals) or if you expect to receive more training data in the future that you want to be able to quickly incorporate into your model.

* Advantages of Decision Trees:

Easy to interpret and explain (for some people -- I'm not sure I fall into this camp). Non-parametric, so you don't have to worry about outliers or whether the data is linearly separable (e.g., decision trees easily take care of cases where you have class A at the low end of some feature x, class B in the mid-range of feature x, and A again at the high end). Their main disadvantage is that they easily overfit, but that's where ensemble methods like random forests (or boosted trees) come in. Plus, random forests are often the winner for lots of problems in classification (usually slightly ahead of SVMs, I believe), they're fast and scalable, and you don't have to worry about tuning a bunch of parameters like you do with SVMs, so they seem to be quite popular these days.

* Advantages of SVMs:

High accuracy, nice theoretical guarantees regarding overfitting, and with an appropriate kernel they can work well even if you're data isn't linearly separable in the base feature space. Especially popular in text classification problems where very high-dimensional spaces are the norm. Memory-intensive and kind of annoying to run and tune, though, so I think random forests are starting to steal the crown.
