<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [From Logistic regression to SVM](#from-logistic-regression-to-svm)
  - [Linear SVM Object/Loss Function](#linear-svm-objectloss-function)
    - [Large margin](#large-margin)
    - [Derived to SVM loss function](#derived-to-svm-loss-function)
    - [Soft margin](#soft-margin)
    - [Objective Function: Primary Form](#objective-function-primary-form)
    - [Dual Form](#dual-form)
      - [Why we use dual form for SVM](#why-we-use-dual-form-for-svm)
      - [Dual Form in Kernel](#dual-form-in-kernel)
    - [Conclusion](#conclusion)
  - [SVM decision boundary](#svm-decision-boundary)
- [kernel](#kernel)
  - [Non-linear decision boundary](#non-linear-decision-boundary)
  - [Kernels:](#kernels)
- [SVM in practice](#svm-in-practice)
  - [General implementation guideline](#general-implementation-guideline)
  - [Overcome Overfitting](#overcome-overfitting)
  - [Logistic Regression and SVM](#logistic-regression-and-svm)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# From Logistic regression to SVM

* Step from Logistic regression to SVM

We want to modify the cost function of logistic regression
![modified cost function](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm0.png)

This is called __hinge loss__

  * We replace the first and second terms of logistic regression with the respective cost functions
  * We remove (1 / m) because it does not matter
  * Instead of A + λB, we use CA + B
  * Parameter C similar to the role (1 / λ)
  * When C = (1 / λ), the two optimization equations would give same parameters θ
![Logistic regression to SVM](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_1.png)

## Linear SVM Object/Loss Function
* Compared to logistic regression,
  * it does not output a probability
  * We get a direct prediction of 1 or 0 instead
    * If θTx is => 0: hθ(x) = 1
    * If θTx is <= 0: hθ(x) = 0

![SVM cost function](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm2.png)

### Large margin
we can see from logistic regression to SVM. the decision margin is large

![SVM large margin](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm4.png)

Here is a example how the large margin looks likely
![large margin example](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm5.png)

* If C is huge, we would want A(the first half) = 0 to minimize the cost function
  * If y = 1: A = 0 such that θTx >= 1
  * If y = 0: A = 0 such that θTx <= -1
* our optimization problem boils down to minimizing the later term only

### Derived to SVM loss function
```math
Minimize ||w||^2, subject to:

(w·xi +b)≥1, if yi =1
(w·xi +b)≤−1, if yi =−1
```


> The last two constraints can be compacted to:

```math
yi(w·xi +b)≥1
```

> This is a quadratic program

### Soft margin
For the very high dimensional problems, sometimes the data are linearly separable. But in the general case they are not, and even if they are, we might prefer a solution that better separates the bulk of the data while ignoring a few weird noise documents

we introduce software variable __ξi, slack__, here is illustration

* Minimize  
```Math
yi(w·xi+b)≥1−ξi, ξi≥0
```


![SVM Objective ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_soft_margin_show.jpg)


### Objective Function: Primary Form
![SVM Objective ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_primary_form.jpg)

* For very large values of the hyper-parameter C, this expression minimizes ∥w∥^2 under the constraint that all training examples are correctly classified with a margin.

* Large C: Low bias, High Variance. Small C, high bias, low variance

* If C is infinite large, it is hard margin

* Smaller values of C relax this constraint and produce markedly better results on noisy problems(addressing overfitting)

This below picture also shows how SVM objective function changed from hard margin to soft margin

![SVM Objective ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_soft_margin.png)

### Dual Form

Solving the primal problem, we obtain the optimal w, but know nothing about the αi. In order to classify a query point x, we need to explicitly compute the scalar product __wTx__
, which may be expensive if d is large.

The Representer Theorem states that the solution w can always be written as a linear combination of the training data:


Solving the dual problem, we obtain the __αi__(where αi=0 for all but a few points - the support vectors). In order to classify a query point x we calculate

![SVM Dual Form ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_dual_form.png)

#### Why we use dual form for SVM

referring http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf

* Assume N is number of training points, and d is dimension of feature vector x.

* Need to learn d parameters for primal, and N for dual, If N <<d then more efficient to solve for α than w. we have to calculate a quantity that depends only on the inner product between x and the points in the training set.

* Moreover, we saw earlier that the αi’s will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we really need to find only the inner products between x and the support vectors

#### Dual Form in Kernel

If we apply kernel, dual form is also better, suppose we map original space d to a higher dimensional space D

__Primal form will solve__
![SVM Primary kernel ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_primary_kernel.png)

__Dual form will solve__
![SVM Primary kernel ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_dual_kernel.png)

> Why don't we use dual form for logistic regression?

since logitic regression dose not have only support vector points, it is
![LR dual ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/lr_dual.png)

Primal form only has __w__, but dual form we need to store __a_i__ and __x__, so data is large



### Conclusion
![SVM conclusion ](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_object_all.jpg)


## SVM decision boundary

more from http://www.ritchieng.com/machine-learning-svms-support-vector-machines/

# kernel

## Non-linear decision boundary

Given the data, is there a different or better choice of the features f1, f2, f3 … fn?
We also see that using high order polynomials is computationally expensive

## Kernels:

> Linear: K(x,y)=xTy

> Polynomial: K(x,y)=(xTy+1)d

> Sigmoid: K(x,y)=tanh(axTy+b)

> RBF: K(x,y)=exp(−γ∥x−y∥2)

# SVM in practice

## General implementation guideline

* We would normally use an SVM software package (liblinear, libsvm etc.) to solve for the parameters θ

* You need to specify the following
  * Choice of parameter C
  * Choice of kernel (similarity function)
    * No kernel is essentially “linear kernel”
      Predict “y = 1” if θ_transpose * x >= 0
      Use this when n is large (number examples) & m is small(feature space)
    * Gaussian kernel
      For this kernel, we have to choose σ^2
      Use this when n is small (number of examples) and/or m is large

## Overcome Overfitting

In practice, the reason that SVMs tend to be resistant to over-fitting, even in cases where the number of attributes is greater than the number of observations, is that it uses regularization.

* They key to avoiding over-fitting lies in careful tuning of the regularization parameter, __C__, the missclasification penalty

* in the case of non-linear SVMs, careful choice of kernel and tuning of the kernel parameters. If you are using a RBF kernel you can try with a wider value for sigma.

## Logistic Regression and SVM

![SVM_LR](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/svm_lr.png)

* The key thing to note is that if there is a huge number of training examples, a Gaussian kernel takes a long time
* The optimization problem of an SVM is a convex problem, so you will always find the global minimum
  * Neural Network: non-convex, may find local optima
