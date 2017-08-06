<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Model selection and data prepare](#model-selection-and-data-prepare)
  - [SoftMax and cross-entropy](#softmax-and-cross-entropy)
    - [SoftMax](#softmax)
    - [Cross-Entropy](#cross-entropy)
- [Bias vs Variance](#bias-vs-variance)
  - [concept](#concept)
  - [Learn from learning curves](#learn-from-learning-curves)
  - [Debug learning algorithm](#debug-learning-algorithm)
    - [Diagnosing Neural Networks](#diagnosing-neural-networks)
    - [Model Complexity Effects:](#model-complexity-effects)
- [Overfitting](#overfitting)
  - [How to overcome overfitting](#how-to-overcome-overfitting)
- [Dimension Reduction](#dimension-reduction)
  - [PCA](#pca)
- [Hyperparameter](#hyperparameter)
  - [Grid search](#grid-search)
  - [Bayesian Optimization for Hyperparameter Tuning](#bayesian-optimization-for-hyperparameter-tuning)
    - [general idea](#general-idea)
    - [Process](#process)
- [Measurement](#measurement)
  - [ROC](#roc)
  - [AUC(Area Under the Curve)](#aucarea-under-the-curve)
- [Feature Selection](#feature-selection)
- [Advantages of some particular algorithms](#advantages-of-some-particular-algorithms)
  - [Advantages of Naive Bayes:](#advantages-of-naive-bayes)
  - [Advantages of Logistic Regression:](#advantages-of-logistic-regression)
  - [Advantages of Decision Trees:](#advantages-of-decision-trees)
  - [Advantages of SVMs:](#advantages-of-svms)
- [Debug the Machine Learning system](#debug-the-machine-learning-system)
  - [Improve accuracy](#improve-accuracy)
  - [37 Reasons why your Neural Network is not working](#37-reasons-why-your-neural-network-is-not-working)
    - [Start Point](#start-point)
    - [Dataset issues](#dataset-issues)
    - [Data Normalization/Augmentation issues](#data-normalizationaugmentation-issues)
    - [Implementation issues](#implementation-issues)
    - [Training issues](#training-issues)

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


Another way to get the loss function of logistic regression

![LR_loss](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lr_loss.png)

![LR_loss_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lr_loss_2.png)

![LR_loss_3](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lr_loss_3.png)

# Bias vs Variance

## concept
* Error due to Bias:

The error due to bias is taken as the difference between the expected (or average) prediction of our model and the correct value which we are trying to predict.

* Error due to Variance:

The error due to variance is taken as the variability of a model prediction for a given data point. Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model


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

# Dimension Reduction

## PCA
* Standardize the data/normalization
* Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
  * covariance:
  ![covariance](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/covariance.png)
* Sort eigenvalues in descending order and choose the k eigenvectors that correspond to the k largest eigenvalues where k is the number of dimensions of the new feature subspace (k≤d).
* Construct the projection matrix W from the selected k eigenvectors.
* Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.

# Hyperparameter

https://jmhessel.github.io/Bayesian-Optimization/

## Grid search
In grid search, we try a set of configurations of hyperparameters and train the algorithm accordingly, choosing the hyperparameter configuration that gives the best performance. In practice, practitioners specify the bounds and steps between values of the hyperparameters, so that it forms a grid of configurations.

Grid search is a costly approach. Assuming we have n hyperparameters and each hyperparameter has two values, then the total number of configurations is 2^{n}. Therefore it is only feasible to do grid search on a small number of configurations.

## Bayesian Optimization for Hyperparameter Tuning

### general idea
https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning/

At a high level, Bayesian Optimization offers a principled method of hyperparameter searching that takes advantage of information one learns during the optimization process. The idea is as follows: you pick some prior belief (it is Bayesian, after all) about how your parameters behave, and search the parameter space of interest by enforcing and updating that prior belief based on your ongoing measurements.

Slightly more specifically, Bayesian Optimization algorithms place a Gaussian Process (GP) prior on your function of interest (in this case, the function that maps from parameter settings to validation set performance). By assuming this prior and updating it after each evaluation of parameter settings, one can infer the shape and structure of the underlying function one is attempting to optimize.


The fundamental question of Bayesian Optimization is: given this set of evaluations, where should we look next? The answer is not obvious. We are fairly confident about the value of the function in regions close to where we evaluate, so perhaps it is best to evaluate in these regions? On the other hand, the unknown function may have a smaller value we’d like to find further away from the region we’ve already explored, though we are less confident in that expectation, as illustrated by the wide confidence intervals.

In practice, different algorithms answer this question differently (based on their so-called acquisition function) but all address the balance between exploration (going to new, unknown regions of parameter space) and exploitation (optimizing within regions where you have higher confidence) to determine good settings of hyperparameters.

### Process
repeat the following:

![Bayesian_Optimization](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/Bayesian_Optimization.png)

# Measurement

## ROC
The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR)

![ROC](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ROC.png)

* True positive rate (TPR):

aka. sensitivity, hit rate, and recall, which is defined as __TP/(TP+FN)__
. Intuitively this metric corresponds to the proportion of positive data points that are correctly considered as positive, with respect to all positive data points. In other words, the higher TPR, the fewer positive data points we will miss.

* False positive rate (FPR), aka. fall-out, which is defined as __FP/(FP+TN)__

. Intuitively this metric corresponds to the proportion of negative data points that are mistakenly considered as positive, with respect to all negative data points. In other words, the higher FPR, the more negative data points we will missclassified.


## AUC(Area Under the Curve)

After we get ROC metrics, we cna plot the ROC curve, and the AUC is like

![AUC](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/AUC.png)

# Feature Selection

# Advantages of some particular algorithms

## Advantages of Naive Bayes:

Super simple, you're just doing a bunch of counts. If the NB __*conditional independence assumption actually holds*__, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn't hold, a NB classifier still often performs surprisingly well in practice. A good bet if you want to do some kind of semi-supervised learning, or want something embarrassingly simple that performs pretty well.

## Advantages of Logistic Regression:

Lots of ways to regularize your model, and you don't have to worry as much about your features being correlated, like you do in Naive Bayes. You also have a nice probabilistic interpretation, unlike decision trees or SVMs, and you can easily update your model to take in new data (using an online gradient descent method), again unlike decision trees or SVMs. Use it if you want a probabilistic framework (e.g., to easily adjust classification thresholds, to say when you're unsure, or to get confidence intervals) or if you expect to receive more training data in the future that you want to be able to quickly incorporate into your model.

## Advantages of Decision Trees:

Easy to interpret and explain (for some people -- I'm not sure I fall into this camp). Non-parametric, so you don't have to worry about outliers or whether the data is linearly separable (e.g., decision trees easily take care of cases where you have class A at the low end of some feature x, class B in the mid-range of feature x, and A again at the high end). Their main disadvantage is that they easily overfit, but that's where ensemble methods like random forests (or boosted trees) come in. Plus, random forests are often the winner for lots of problems in classification (usually slightly ahead of SVMs, I believe), they're fast and scalable, and you don't have to worry about tuning a bunch of parameters like you do with SVMs, so they seem to be quite popular these days.

## Advantages of SVMs:

High accuracy, nice theoretical guarantees regarding overfitting, and with an appropriate kernel they can work well even if you're data isn't linearly separable in the base feature space. Especially popular in text classification problems where very high-dimensional spaces are the norm. Memory-intensive and kind of annoying to run and tune, though, so I think random forests are starting to steal the crown.

# Debug the Machine Learning system

## Improve accuracy

current model is 85%, what could be next step to increase to 90%.

* data: is training representing all distribution enough? is data enough?
* model: overfitting


## 37 Reasons why your Neural Network is not working

https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

### Start Point

* Start with a simple model that is known to work for this type of data (for example, VGG for images). Use a standard loss if possible.
* Turn off all bells and whistles, e.g. regularization and data augmentation.
* If fine tuning a model, double check the preprocessing, for it should be the same as the original model’s training.
* Verify that the input data is correct.
* Start with a really small dataset (2–20 samples). Overfit on it and gradually add more data.
* Start gradually adding back all the pieces that were omitted: augmentation/regularization, custom loss functions, try more complex models.

### Dataset issues

* Examine Input

  * Check your input data

  I’ve more than once mixed the width and the height of an image. Sometimes, I would feed all zeroes by mistake. Or I would use the same batch over and over. So print/display a couple of batches of input and target output and make sure they are OK.

  * Try random input

  Try passing random numbers instead of actual data and see if the error behaves the same way. If it does, it’s a sure sign that your net is turning data into garbage at some point. Try debugging layer by layer /op by op/ and see where things go wrong.

  * Check the data loader

  * Is the relationship between input and output too random?

  Maybe the non-random part of the relationship between the input and output is too small compared to the random part (one could argue that stock prices are like this). I.e. the input are not sufficiently related to the output. There isn’t an universal way to detect this as it depends on the nature of the data.

  * Is there too much noise in the dataset?

  Check a bunch of input samples manually and see if labels seem off.

* Input data process

  * Shuffle the dataset

  If your dataset hasn’t been shuffled and has a particular order to it (ordered by label) this could negatively impact the learning.

  * Reduce class imbalance

  * Do you have enough training examples?

  If you are training a net from scratch (i.e. not finetuning), you probably need lots of data. For image classification, people say you need a 1000 images per class or more.

* Examine the batch

  * Make sure your batches don’t contain a single label

  This can happen in a sorted dataset (i.e. the first 10k samples contain the same class). Easily fixable by shuffling the dataset.

  * Reduce batch size

  This paper points out that having a very large batch can reduce the generalization ability of the model.

### Data Normalization/Augmentation issues

* Standardize the features(Normalization)

Did you standardize your input to have zero mean and unit variance?

* Check the preprocessing of your pretrained model

If you are using a pretrained model, make sure you are using the same normalization and preprocessing as the model was when training. For example, should an image pixel be in the range [0, 1], [-1, 1] or [0, 255]?

* Check the preprocessing for train/validation/test set

CS231n points out a common pitfall:
“… any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation/test data. E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting the data into train/val/test splits would be a mistake. “
Also, check for different preprocessing in each sample or batch.


### Implementation issues

* Try solving a simpler version of the problem

This will help with finding where the issue is. For example, if the target output is an object class and coordinates, try limiting the prediction to object class only


* Look for correct loss “at chance”

Again from the excellent CS231n: Initialize with small parameters, without regularization. For example, if we have 10 classes, at chance means we will get the correct class 10% of the time, and the Softmax loss is the negative log probability of the correct class so: -ln(0.1) = 2.302.
After this, try increasing the regularization strength which should increase the loss.

* Check the Loss function
  * Check your loss function for any potential bug
  * Verify loss input
  * Adjust loss weights.

  If your loss is composed of several smaller loss functions, make sure their magnitude relative to each is correct.

* Increase network size
Maybe the expressive power of your network is not enough to capture the target function. Try adding more layers or more hidden units in fully connected layers.
* Check for hidden dimension errors
If your input looks like (k, H, W) = (64, 64, 64) it’s easy to miss errors related to wrong dimensions. Use weird numbers for input dimensions (for example, different prime numbers for each dimension) and check how they propagate through the network.
* Explore Gradient checking

### Training issues

* Solve for a really small dataset

Overfit a small subset of the data and make sure it works. For example, train with just 1 or 2 examples and see if your network can learn to differentiate these. Move on to more samples per class.

* Check weights initialization

If unsure, use Xavier or He initialization. Also, your initialization might be leading you to a bad local minimum, so try a different initialization and see if it helps.

* Change your hyperparameters
Maybe you using a particularly bad set of hyperparameters. If feasible, try a grid search or bayesian-optimization-hyperparameter-tuning

* Reduce regularization

Too much regularization can cause the network to underfit badly. Reduce regularization such as dropout, batch norm, weight/bias L2 regularization, etc. In the excellent [Practical Deep Learning for coders](http://course.fast.ai) course, Jeremy Howard advises getting rid of underfitting first. This means you overfit the training data sufficiently, and only then addressing overfitting.

* Visualize the training

  * Monitor the activations, weights, and updates of each layer. Make sure their magnitudes match. For example, the magnitude of the updates to the parameters (weights and biases) should be 1-e3.
  * Consider a visualization library like Tensorboard and Crayon. In a pinch, you can also print weights/biases/activations.
  * Be on the lookout for layer activations with a mean much larger than 0. Try Batch Norm or ELUs.
  * Deeplearning4j points out what to expect in histograms of weights and biases:

* Try a different optimizer

Your choice of optimizer shouldn’t prevent your network from training unless you have selected particularly bad hyperparameters. However, the proper optimizer for a task can be helpful in getting the most training in the shortest amount of time. The paper which describes the algorithm you are using should specify the optimizer. If not, I tend to use Adam or plain SGD with momentum, see great [post](http://ruder.io/optimizing-gradient-descent/)
