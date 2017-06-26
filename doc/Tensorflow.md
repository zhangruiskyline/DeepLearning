<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [A SIMPLE TENSORFLOW PROGRAM](#a-simple-tensorflow-program)
  - [Sessions](#sessions)
  - [Placeholder](#placeholder)
  - [Graph Element](#graph-element)
  - [How to run](#how-to-run)
- [A simple predict model](#a-simple-predict-model)
  - [Variable](#variable)
  - [Optimization](#optimization)
- [Checkpointing and Reusing TensorFlow Models](#checkpointing-and-reusing-tensorflow-models)
  - [Saver object](#saver-object)
- [Keras Usage](#keras-usage)
  - [Merge](#merge)
- [Batch Normalization](#batch-normalization)
  - [Intention](#intention)
  - [Process](#process)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# A SIMPLE TENSORFLOW PROGRAM

based on https://nathanbrixius.wordpress.com/2016/05/18/an-introduction-to-tensorflow/

I think the TensorFlow tutorials are too complicated for a beginner, so I’m going to present a simple TensorFlow example that takes input x, adds one to it, and stores it in an output array y. Many TensorFlow programs, including this one, have four distinct phases:

1. Create TensorFlow objects that model the calculation you want to carry out,
2. Get the input data for the model,
3. Run the model using the input data,
4. Do something with the output.

```python
import numpy as np
import tensorflow as tf
import math

def add_one():
with tf.Session() as session:
# (1)
x = tf.placeholder(tf.float32, [1], name=’x’) # fed as input below
y = tf.placeholder(tf.float32, [1], name=’y’) # fetched as output below
b = tf.constant(1.0)
y = x + b # here is our ‘model’: add one to the input.

        x_in = [2] # (2)
y_final = session.run([y], {x: x_in}) # (3)
print(y_final) # (4)

```

## Sessions

Sessions contain “computational graphs” that represent calculations to be carried out. In our example, we want to create a computational graph that represents adding the constant 1.0 to an input array x.

## Placeholder

 A placeholder is an interface between a computational graph element and your data. Placeholders can represent input or output, and in my case  x represents the value to send in, and y represents the result. The second argument of the placeholder function is the shape of the placeholder, which is a single dimensional Tensor with one entry. You can also provide a name, which is useful for debugging purposes.

## Graph Element

The next line, __y = x + b__, is the computational model we want TensorFlow to calculate. This line does not actually compute anything, even though it looks like it should. It simply creates data structures (called “graph elements”) that represent the addition of x and b, and the assignment of the result to the placeholder y. Each of the items in my picture above is a graph element. These graph elements are processed by the TensorFlow engine when Session.run() is called. Part of the magic of TensorFlow is to efficiently carry out graph element evaluation, even for very large and complicated graphs.

## How to run

Now that the model is created, we turn to assembling the input and running the model. Our model has one input x, so we create a list x_in that will be associated with the placeholder x. If you think of a TensorFlow model as a function in your favorite programming language, the placeholders are the arguments. Here we want to “pass” x_in as the value for the “parameter” x. This is what happens in the __session.run()__ call. The first argument is a list of graph elements that you would like TensorFlow to evaluate. In this case, we’re interested in evaluating the output placeholder y, so that’s what we pass in. Session.run will return an output value for each graph element that you pass in as the first argument, and the value will correspond to the evaluated value for that element. In English this means that y_final is going to be an array that has the result: x + 1. The second argument to run is a dictionary that specifies the values for input placeholders. This is where we associate the input array x_in with the placeholder x.

When __Session.run()__ is called, TensorFlow will determine which elements of the computational graph need to be evaluated based on what you’ve passed in. It will then carry out the computations and then bind result values accordingly. The final line prints out the resulting array.

# A simple predict model
This model will fit a line y = m * x + b to a series of points (x_i, y_i). This code is not the best way fit a line

There are two functions below: one to generate test data, and another to create and run the TensorFlow model:

```python
import TensorFlow as tf

def make_data(n):
    np.random.seed(42) # To ensure same data for multiple runs
    x = 2.0 * np.array(range(n))
    y = 1.0 + 3.0 * (np.array(range(n)) + 0.1 * (np.random.rand(n) – 0.5))
    return x, y

def fit_line(n=1, log_progress=False):
    with tf.Session() as session:
        x = tf.placeholder(tf.float32, [n], name=’x’)
        y = tf.placeholder(tf.float32, [n], name=’y’)
        m = tf.Variable([1.0], trainable=True) # training variable: slope
        b = tf.Variable([1.0], trainable=True) # training variable: intercept
        y = tf.add(tf.mul(m, x), b) # fit y_i = m * x_i + b

        # actual values (for training)
        y_act = tf.placeholder(tf.float32, [n], name=’y_’)

        # minimize sum of squared error between trained and actual.
        error = tf.sqrt((y – y_act) * (y – y_act))
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)
        train_step = tf.train.AdamOptimizer(0.05).minimize(error)

        # generate input and output data with a little random noise.
        x_in, y_star = make_data(n)

        init = tf.initialize_all_variables()
        session.run(init)
        feed_dict = {x: x_in, y_act: y_star}
        for i in range(30 * n):
            y_i, m_i, b_i, _ = session.run([y, m, b, train_step], feed_dict)
            err = np.linalg.norm(y_i – y_star, 2)
            if log_progress:
                print(“%3d | %.4f %.4f %.4e” % (i, m_i, b_i, err))

        print(“Done! m = %f, b = %f, err = %e, iterations = %d”
              % (m_i, b_i, err, i))
        print(”      x: %s” % x_in)
        print(“Trained: %s” % y_i)
        print(” Actual: %s” % y_star)

```

After we create a TensorFlow session, our next two steps are to create placeholders for our input x and output y, similar to our first example. These are both Tensors of size n since that’s how many data points we have.

## Variable

The next line creates a TensorFlow variable to represent the slope m. A variable is a value that is retained between calls to Session.run(). If the value is an input or an output from the model, we don’t want a variable – we want a placeholder. If the value remains constant during our computation, we don’t want a variable – we want a __tf.constant__. We want variables when we want TensorFlow to train the value based on some criteria in our model.

Notice when we create the Variable objects we supply __initial values__ for the variable, and a __“trainable”__ flag. Providing TensorFlow with initial values for a variable informs TensorFlow of the dimensionality and type – in our case m and b are single dimensional Tensors of size 1, but they could just as easily be multidimensional and/or integer.

![computation graph](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/tensorflowgraph_1.png)

## Optimization

Here comes the next new concept: optimization. We have specified our model, and the error in the model, but now we want TensorFlow to find the best possible values for m and b given the error expression. Optimization is carried out in TensorFlow by repeatedly calling __Session.run()__ with an Optimization object “fed” as input. An Optimization carries out logic that adjusts the variables in a way that will hopefully improve the value of the error expression. In our case we will use an [AdamOptimizer](https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#AdamOptimizer) object.

The parameter to AdamOptimizer controls how much the optimizer adjusts the variables on each call – larger is more aggressive. All Optimizer objects have a __minimize()__ method that lets you pass in the expression you want to optimize. You can see that the train_step, the value returned by the AdamOptimizer, is passed into the __Session.run()__ call.

The key ingredient is the gradient of the variables that you are trying to optimize. TensorFlow computes gradients by creating computational graph elements for the gradient expressions and evaluating them.

TensorFlow can do this because it has a *symbolic* representation of the expressions(see [this link](https://stackoverflow.com/questions/36370129/does-tensorflow-use-automatic-or-symbolic-gradients)) you’re trying to compute (it’s in the picture above). Since a call to an optimizer is a single step, __Session.run()__ must be called repeatedly in a loop to get suitable values.

# Checkpointing and Reusing TensorFlow Models

As we discussed last time, training a model means finding variable values that suit a particular purpose, for example finding a slope and intercept that defines a line that best fits a series of points. Training a model can be computationally expensive because we have to search for the best variable values through optimization. Suppose we want to use the results of this trained model over and over again, but without re-training the model each time. You can do this in TensorFlow using the Saver object.

## Saver object

* Creating a Saver and telling the Saver which variables you want to save,
* Save the variables to a file,
* Restore the variables from a file when they are needed.

> A Saver deals only with Variables. It does not work with placeholders, sessions, expressions, or any other kind of TensorFlow object. Here is a simple example that saves and restores two variables:


```python
import tensorflow as tf

def save(checkpoint_file=’hello.chk’):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name=’x’)
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name=’y’)
        not_saved = tf.Variable([-1, -2], name=’not_saved’)
        session.run(tf.initialize_all_variables())

        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([x, y])
        saver.save(session, checkpoint_file)

def restore(checkpoint_file=’hello.chk’):
    x = tf.Variable(-1.0, validate_shape=False, name=’x’)
    y = tf.Variable(-1.0, validate_shape=False, name=’y’)
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.all_variables()))

def reset():
    tf.reset_default_graph()

```

When you create a Saver, you should specify a list (or dictionary) of Variable objects you wish to save. (If you don’t, TensorFlow will assume you are interested in all the variables in your current session.) The shapes and values of these values will be stored in binary format when you call the save() method, and retrieved on restore(). Notice in my last function, when I create x and y, I give dummy values and say validate_shape=False. This is because I want the saver to determine the values and shapes when the variables are restored. If you’re wondering why the reset() function is there, remember that computational graphs are associated with Sessions. I want to “clear out” the state of the Session so I don’t have multiple x and y objects floating around as we call save and restore().

If you want to do anything useful with the Variables you restore, you may need to recreate the rest of the computational graph.
* The computational graph that you use with restored Variables need not be the same as the one that you used when saving. That can be useful!
* Saver has additional methods that can be helpful if your computation spans machines, or if you want to avoid overwriting old checkpoints on successive calls to save().

Here is a optional save method:

```python
fit_line(5, checkpoint_file=’vars.chk’)
reset()
fit_line(5, checkpoint_file=’vars.chk’, restore=True)

```

With this version, I could easily “score” new data points x using my trained model.

```python
def fit_line(n=1, log_progress=False, iter_scale=200,
             restore=False, checkpoint_file=None):
    with tf.Session() as session:
        x = tf.placeholder(tf.float32, [n], name=’x’)
        y = tf.placeholder(tf.float32, [n], name=’y’)
        m = tf.Variable([1.0], name=’m’)
        b = tf.Variable([1.0], name=’b’)
        y = tf.add(tf.mul(m, x), b) # fit y_i = m * x_i + b
        y_act = tf.placeholder(tf.float32, [n], name=’y_’)

        # minimize sum of squared error between trained and actual.
        error = tf.sqrt((y – y_act) * (y – y_act))
        train_step = tf.train.AdamOptimizer(0.05).minimize(error)

        x_in, y_star = make_data(n)

        saver = tf.train.Saver()
        feed_dict = {x: x_in, y_act: y_star}
        if restore:
            print(“Loading variables from ‘%s’.” % checkpoint_file)
            saver.restore(session, checkpoint_file)
            y_i, m_i, b_i = session.run([y, m, b], feed_dict)
        else:
            init = tf.initialize_all_variables()
            session.run(init)
            for i in range(iter_scale * n):
                y_i, m_i, b_i, _ = session.run([y, m, b, train_step],
                                               feed_dict)
                err = np.linalg.norm(y_i – y_star, 2)
                if log_progress:
                    print(“%3d | %.4f %.4f %.4e” % (i, m_i, b_i, err))

            print(“Done training! m = %f, b = %f, err = %e, iter = %d”
                  % (m_i, b_i, err, i))
            if checkpoint_file is not None:
                print(“Saving variables to ‘%s’.” % checkpoint_file)
                saver.save(session, checkpoint_file)

        print(”      x: %s” % x_in)
        print(“Trained: %s” % y_i)
        print(” Actual: %s” % y_star)

```

# Keras Usage

## Merge

```python
from keras.layers import Input, merge
from keras.models import Model
import numpy as np

input_a = np.reshape([1, 2, 3], (1, 1, 3))
input_b = np.reshape([4, 5, 6], (1, 1, 3))

print(input_a)
print(input_b)

a = Input(shape=(1, 3))
b = Input(shape=(1, 3))


concat = merge([a, b], mode='concat', concat_axis=-1)
dot = merge([a, b], mode='dot', dot_axes=2)
cos = merge([a, b], mode='cos', dot_axes=2)

model_concat = Model(input=[a, b], output=concat)
model_dot = Model(input=[a, b], output=dot)
model_cos = Model(input=[a, b], output=cos)


print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_a, input_b]))
print(model_cos.predict([input_a, input_b]))

```

Output will be

```python
[[[1 2 3]]]
[[[4 5 6]]]
[[[ 1.  2.  3.  4.  5.  6.]]]
[[[ 32.]]]
[[[[ 0.97463191]]]]
```

# Batch Normalization

[The paper](https://arxiv.org/abs/1502.03167): Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

Batch Normalization is a technique to provide any layer in a Neural Network with inputs that are zero mean/unit variance.

This is great [blog](http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) to explain Batch Normalization

## Intention
Batch normalization (BN) solves a problem called internal covariate shift. Covariate shift means the distribution of the features is different in different parts of the training/test data, breaking the i.i.d assumption used across most of ML.

Internal covariate shift refers to covariate shift occurring within a neural network, i.e. going from (say) layer 2 to layer 3. This happens because, as the network learns and the weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning.

For deep network, if only input layer and output layer are whitened, during training, hidden layers are gradually deviated from zero mean,unit variance and decorrelated conditions, which is called internal co-variate shift, as shown below

![internal_co-variate_ shift](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/internal_co-variate_ shift.png)


## Process

BN helps by making the data flowing between intermediate layers of the network look like whitened data.

* Learning faster: Learning rate can be increased compare to non-batch-normalized version.
Increase Accuracy: Flexibility on mean and variance value for every dimension in every hidden layer provides better learning, hence accuracy of the network.
Normalization or Whitening of the inputs to each layer: Zero means, unit variances and or not decorrelated.
* To remove the ill-effect of Internal Covariate shift:Transformation makes data to big or to small; change of the input distribution away from normalization due to successive transformation.
* Not-Stuck in the saturation mode: Even if ReLU is not used.
* Integrate Whitening within the gradient descent optimization: Decoupled Whitening between training steps, which modifies network directly, reduces the effort of optimization. So, model blows up when normalization parameters are computed outside the gradient descent step.
Whitening within gradient descent: Requires inverse square root of covariance matrix as well as derivatives for backpropagation
* Normalization of mini-batch: Estimation of mean and variance are computed after each mini-batch rather than entire training set. Even ignoring the joint covariance as it will create singular co-variance matrices for such small number of training sample per mini-batch compare to high dimension size of the hidden layer.
* Learning of scale and shift for every dimension: Scaled and shifted values are passed to the next layer, whether mean and variances are calculated after getting all mini-batch activation of current layer. So, forward pass of all the samples within the mini-batch should pass layer wise. Backpropagation is required for getting gradient of weights as well as scaling (variance) and shift (mean).
* Inference: During inference moving averaged mean and variance parameters during mini batch training are considered.
* Convolution Neural Network: Whitening of intermediate layers, before or after the nonlinearity creates a lot of new innovation pathways [11-15].
