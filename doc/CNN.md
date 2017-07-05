<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ConvNet Architectures](#convnet-architectures)
  - [Example Architecture:](#example-architecture)
  - [Spatial arrangement.](#spatial-arrangement)
  - [Parameter sharing](#parameter-sharing)
  - [Summary](#summary)
  - [Layered pattern](#layered-pattern)
  - [Parameter Size Analysis](#parameter-size-analysis)
- [CNN Research Milestone](#cnn-research-milestone)
  - [AlexNet](#alexnet)
  - [VGG](#vgg)
  - [Google LeNet(Inception)](#google-lenetinception)
  - [ResNet(By Microsoft)](#resnetby-microsoft)
  - [Region Based CNNs (R-CNN - 2013, Fast R-CNN - 2015, Faster R-CNN - 2015)](#region-based-cnns-r-cnn---2013-fast-r-cnn---2015-faster-r-cnn---2015)
- [Code Examples](#code-examples)
  - [Keras Intro on how to train CNN](#keras-intro-on-how-to-train-cnn)
  - [Keras on Imagenet](#keras-on-imagenet)
- [Transfer Learning](#transfer-learning)
  - [Example: fully connected Resnet to Convolutional based](#example-fully-connected-resnet-to-convolutional-based)
    - [Changes made:](#changes-made)
    - [get similar image](#get-similar-image)
- [Data Augmentation](#data-augmentation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# ConvNet Architectures

http://cs231n.github.io/convolutional-networks/

We use three main types of layers to build ConvNet architectures: *Convolutional Layer*, *Pooling Layer*, and *Fully-Connected Layer* (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.

## Example Architecture:

Overview. We will go into more details below, but a simple ConvNet for CIFAR-10 classification could have the architecture __*[INPUT - CONV - RELU - POOL - FC]*__. In more detail:

* INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
* CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
* RELU layer will apply an elementwise activation function, This leaves the size of the volume unchanged ([32x32x12]).
* POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
* FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

Example 1. For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of *5x5x3* = 75 weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

Example 2. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of *3x3x20* = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

## Spatial arrangement.

Three hyperparameters control the size of the output volume: *the depth*, *stride* and *zero-padding*. We discuss these next:
> depth

* First, the __depth__ of the output volume is a *hyperparameter*: it corresponds to the __number of filters__ we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a __depth column__ (some people also prefer the term fibre).

> stride

* Second, we must specify the __stride__ with which we slide the filter.

> padding

* It will be convenient to pad the input volume with zeros around the border. The size of this __zero-padding__ is a *hyperparameter*.

We can compute the spatial size of the output volume as a function of the input volume size __*W*__, the receptive field size of the Conv Layer neurons __*F*__, the stride with which they are applied __*S*__, and the amount of zero padding used __*P*__ on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by __*(W+2P-F)/S+1*__.

## Parameter sharing

ImageNet challenge in 2012 accepted images of size [227x227x3]. On the first Convolutional Layer, it used neurons with receptive field size F=11, stride S=4 and no zero padding P=0. Since (227 - 11)/4 + 1 = 55, and since the Conv layer had a depth of K=96, the Conv layer output volume had size [55x55x96].

we see that there are 55 x 55 x 96 = 290,400 neurons in the first Conv Layer, and each has 11 x 11 x 3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high.

* Idea of Parameter sharing: if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2).

In other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96 x 11 x 11 x 3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55x55 neurons in each depth slice will now be using the same parameters.

## Summary

* Accepts a volume of size W1×H1×D1
* Produces a volume of size W2×H2×D2

__W2=(W1−F+2P)/S+1__
__H2=(H1−F+2P)/S+1__

* Where Number of filters K, filter size F, the stride S and amount of zero padding P
* With parameter sharing, it introduces __F*F*D1__ weights per filter, for a total of __(F*F*D1)*K__
wights and __K__ biases.


## Layered pattern

```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

```
where the * indicates repetition, and the POOL? indicates an optional pooling layer. Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). For example, here are some common ConvNet architectures you may see that follow this pattern:


* INPUT -> FC, implements a linear classifier. Here N = M = K = 0.
* INPUT -> CONV -> RELU -> FC
* INPUT -> [CONV -> RELU -> POOL]* 2 -> FC -> RELU -> FC. Here we see that there is a single CONV layer between every POOL layer.
* INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]* 3 -> [FC -> RELU]* 2 -> FC Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

> Stack convolution layers with small filter vs one big filter

First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive.

Second, it saves parameters:
```shell
//assuming we have C channels
C×(7×7×C)=49C^2 for one big 7*7 filter
3×(C×(3×3×C))=27C^2 for 3 stacked 3*3 filters

```


Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters.

## Parameter Size Analysis

For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filter sizes of 11x11 and stride of 4.

Lets break down the VGGNet in more detail as a case study. The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding). We can write out the size of the representation at each step of the processing and keep track of both the representation size and the total number of weights:

```
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000

TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters

```

# CNN Research Milestone

https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

## AlexNet
![AlexNet structure](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/AlexNet.png)


* Used ReLU (Found to decrease training time as ReLUs are several times faster than the conventional tanh function).
* Used data augmentation techniques that consisted of image translations, horizontal reflections, and patch extractions.
* Implemented dropout layers in order to combat the problem of overfitting to the training data.
* Trained the model using batch stochastic gradient descent,

## VGG
![VGG structure](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/VGGNet.png)
* The use of only 3x3 sized filters is quite different from AlexNet’s 11x11 filters in the first layer and ZF Net’s 7x7 filters. The authors’ reasoning is that the combination of two 3x3 conv layers has an effective receptive field of 5x5.

* One of the benefits is a decrease in the number of parameters. Also, with two conv layers, we’re able to use two ReLU layers instead of one.

* 3 conv layers back to back have an effective receptive field of 7x7.

* As the spatial size of the input volumes at each layer decrease (result of the conv and pool layers), the depth of the volumes increase due to the increased number of filters as you go down the network.

* the number of filters doubles after each maxpool layer. This reinforces the idea of shrinking spatial dimensions, but growing depth.

* Worked well on both image classification and localization tasks.

it reinforced the notion that __convolutional neural networks have to have a deep network of layers in order for this hierarchical representation of visual data to work__. Keep it deep. Keep it simple.

## Google LeNet(Inception)
![Google LeNet structure](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/GoogLeNet.png)

This box is called an Inception module.

![Inception structure](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/GoogLeNet2.png)

* The 1x1 convolutions (or network in network layer) provide a method of dimensionality reduction. For example, let’s say you had an input volume of 100x100x60 (This isn’t necessarily the dimensions of the image, just the input to any layer of the network). Applying 20 filters of 1x1 convolution would allow you to reduce the volume to 100x100x20. [1x1 Conv explained](http://iamaaditya.github.io/2016/03/one-by-one-convolution/) or the [udacity link](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/63742133070923)
* To make network deep by adding an “inception module” like Network in Network paper, as described above.
* To reduce the dimensions inside this “inception module”.
* To add more non-linearity by having ReLU immediately after every 1x1 convolution.
* No use of fully connected layers! They use an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of parameters.
* Uses 12x fewer parameters than AlexNet.
* During testing, multiple crops of the same image were created, fed into the network, and the softmax probabilities were averaged to give us the final solution.(Use idea of R-CNN)

## ResNet(By Microsoft)
![Resnet structure](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ResNet.png)
The idea behind a residual block is that you have your input x go through conv-relu-conv series. This will give you some F(x). That result is then added to the original input x. Let’s call that H(x) = F(x) + x. In traditional CNNs, your H(x) would just be equal to F(x) right? So, instead of just computing that transformation (straight from x to F(x)), we’re computing the term that you have to add, F(x), to your input, x. Basically, the mini module shown below is computing a “delta” or a slight change to the original input x to get a slightly altered representation

* “Ultra-deep” – Yann LeCun.152 layers…
* Interesting note that after only the first 2 layers, the spatial size gets compressed from an input volume of 224x224 to a 56x56 volume.
* Authors claim that a naïve increase of layers in plain nets result in higher training and test error (Figure 1 in the paper).

## Region Based CNNs (R-CNN - 2013, Fast R-CNN - 2015, Faster R-CNN - 2015)

The purpose of R-CNNs is to solve the problem of object detection. Given a certain image, we want to be able to draw bounding boxes over all of the objects. The process can be split into two general components, the region proposal step and the classification step.

> R-CNN

The purpose of R-CNNs is to solve the problem of object detection. Given a certain image, we want to be able to draw bounding boxes over all of the objects. The process can be split into two general components, the region proposal step and the classification step.

* Use selective search: Selective Search performs the function of generating 2000 different regions that have the highest probability of containing an object. After we’ve come up with a set of region proposals

* These proposals are then “warped” into an image size that can be fed into a trained CNN(Alex Net), which extracts a feature vector for each region.

* This vector is then used as the input to a set of linear SVMs that are trained for each class and output a classification.

* The vector also gets fed into a bounding box regressor to obtain the most accurate coordinates.

![R-CNN architecture](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rcnn.png)

> Fast-RCNN

RCNN: Training took multiple stages (ConvNets to SVMs to bounding box regressors), was computationally expensive, and was extremely slow.

* Fast R-CNN was able to solve the problem of speed by basically sharing computation of the conv layers between different proposals and swapping the order of generating region proposals and running the CNN.

* Image is first fed through a ConvNet, features of the region proposals are obtained from the last feature map of the ConvNet

* Lastly we have our fully connected layers as well as our regression and classification heads.

![Fast R-CNN architecture](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/FastRCNN.png)

> Faster R-CNN

Faster R-CNN works to combat the somewhat complex training pipeline that both R-CNN and Fast R-CNN exhibited. The authors insert a region proposal network (RPN) after the last convolutional layer. This network is able to just look at the last convolutional feature map and produce region proposals from that. From that stage, the same pipeline as R-CNN is used (ROI pooling, FC, and then classification and regression heads).

![Faster R-CNN architecture](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/FasterRCNN.png)


# Code Examples
## Keras Intro on how to train CNN

[This Blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) will go over the following options:

* training a small network from scratch (as a baseline)
* using the bottleneck features of a pre-trained network
* fine-tuning the top layers of a pre-trained network

This will lead us to cover the following Keras features:

* fit_generator for training Keras a model using Python data generators
* layer freezing and model fine-tuning ...and more.

## Keras on Imagenet

```Python
# import the necessary packages
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])

```

This step isn’t strictly required since Keras provides helper functions to load images (which I’ll demonstrate in the next code block), but there are differences in how both these functions work, so if you intend on applying any type of OpenCV functions to your images, I suggest loading your image via cv2.imread  and then again via the Keras helpers.

```python
# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)

# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
# Another similar way to do so : preprocess_input(image[np.newaxis])
image = preprocess_input(image)

```

If at this stage we inspect the .shape  of our image , you’ll notice the shape of the NumPy array is (3, 224, 224) — each image is 224 pixels wide, 224 pixels tall, and has 3 channels (one for each of the Red, Green, and Blue channels, respectively).

However, before we can pass our image  through our CNN for classification, we need to expand the dimensions to be (1, 3, 224, 224).

Why do we do this?

When classifying images using Deep Learning and Convolutional Neural Networks, we often send images through the network in “batches” for efficiency. Thus, it’s actually quite rare to pass only one image at a time through the network — unless of course, you only have one image to classify.


```python
img = imread(path)
plt.imshow(img)

img = imresize(img, (224,224)).astype("float32")
# add a dimension for a "batch" of 1 image
img_batch = preprocess_input(img[np.newaxis])

predictions = model.predict(img_batch)
print(np.asarray(predictions).shape)
#(1,1000)
decoded_predictions= decode_predictions(predictions)
# decoded_predictions
print(decoded_predictions)
for s, name, score in decoded_predictions[0]:
    print(name, score)

```
Supplying weights="imagenet"  indicates that we want to use the pre-trained ImageNet weights for the respective model.
* prediction: predicted probabilities
Once the network has been loaded and initialized, we can predict class labels by making a call to the .predict  method of the model . These predictions are actually a NumPy array with *1,000 entries* — the __*predicted probabilities*__ associated with each class in the ImageNet dataset.
* decode_prediction
Calling __decode_predictions__  on these predictions gives us the ImageNet Unique ID of the label, along with a human-readable text version of the label(default to return highest 5 results).


# Transfer Learning
* Use pre-trained weights, remove last layers to compute representations of images
* Train a classification model from these features on a new classification task
* The network is used as a generic feature extractor
* Better than handcrafted feature extraction on natural images
* generate similar image recommendations

## Example: fully connected Resnet to Convolutional based
> Goal: Load a CNN model pre-trained on ImageNet
Transform the network into a Fully Convolutional Network
Apply the network perform weak segmentation on images


* Out of the res5c residual block, the resnet outputs a tensor of shape  W×H×2048
* For the default ImageNet input,  224×224, the output size is  7×7×2048
* After this bloc, the ResNet uses an average pooling AveragePooling2D(pool_size=(7, 7)) with (7, 7) strides which divides by 7 the width and height

So the architecture after base model is [link]( https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py)

```
x = base_model.output
x = Flatten()(x)
x = Dense(1000)(x)
x = Softmax()(x)
```

### Changes made:

To keep spatial information as much as possible, we will remove the average pooling.
We want to retrieve the labels information, which is stored in the Dense layer. We will load these weights afterwards
We will change the Dense Layer to a Convolution2D layer to keep spatial information, to output a  W×H×1000.
We can use a kernel size of (1, 1) for that new Convolution2D layer to pass the spatial organization of the previous layer unchanged.
We want to apply a softmax only on the last dimension so as to preserve the  W×H
  spatial information. We build the following Custom Layer to apply a softmax only to the last dimension of a tensor:

### get similar image

```python
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image

model = ResNet50(include_top=True, weights='imagenet')
# Get the representations

input = model.layers[0].input
output = model.layers[-2].output
base_model = Model(input, output)

# Computing representations of all images can be time consuming. This is usually made by large batches on a GPU for massive performance gains.
# For the remaining part, we will use pre-computed representations saved in h5 format
import h5py

# Load pre-calculated representations
h5f = h5py.File('img_emb.h5','r')
out_tensors = h5f['img_emb'][:]  # (504, 2048)


# For example we can use this to recommend most similar images
# use norm factor to get distance between target image id and whole tensor mode
# then sort, which gives us most similar images
def most_similar(idx, top_n=5):
		# out_tensors is (504, 2048),
    dists = np.linalg.norm(out_tensors - out_tensors[idx], axis = 1)
    print(dists.shape)
    sorted_dists = np.argsort(dists)
    return sorted_dists[:top_n]


# extend to classification nearest neighbors
cos_to_item = np.dot(normed_out_tensors, normed_out_tensors[idx])
items = np.where(cos_to_item > 0.54)
print(items)
[display(paths[s]) for s, _ in zip(items[0], range(10))];
```

# Data Augmentation

```python
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    channel_shift_range=9,
    fill_mode='nearest'
)
train_flow = image_gen.flow_from_directory(train_folder)
model.fit_generator(train_flow, train_flow.nb_sample)

```
