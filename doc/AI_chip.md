<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Overview](#overview)
  - [Applications](#applications)
    - [Cost analysis](#cost-analysis)
  - [Deep Learning Break Down to Chip](#deep-learning-break-down-to-chip)
    - [Android on cellphone](#android-on-cellphone)
    - [Performance Analysis:](#performance-analysis)
    - [Scenario analysis](#scenario-analysis)
  - [HW Acceleration](#hw-acceleration)
  - [Deep Learning Accelerator](#deep-learning-accelerator)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Overview

## Applications

Applications: Image, voice

Markets: cellphone, tablet, camera, home, auto, server

### Cost analysis
8 cents per 1mm^2

## Deep Learning Break Down to Chip

* Typical applications: Image(meitu), voice assistant, AR

* Requirements: INT 8 and Float 16

The typical architecture will be as follow

![computation](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/computation.png)

### Android on cellphone

Follow Google Android NN interface: Android interface supports TensorFlow and Tensorflow-Lite, it can support CPU, GPU, DSP and ASIC

![Android_NN](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/Android_NN.png)


### Performance Analysis:

* Under TSMC 16nm and ARM, The big core is about 10-100 Gops/W(INT 8), Little Core is 100G-1T ops/W.

* Cellphone GPU is 300Gops/W, so we need to add HW Accelerator if we want to reach 1Tops/W

### Scenario analysis

The Long time running cellphone power consumption should be less than 2.5W, so Deep Learning can use at most 1.5W. So it will be 1.5T ops(INT)

* Image: Burst type.

* Voice: 100G ops, can be done in little core

* AR: 30-60 fps/s, so CPU can not handle it, we may need GPU Acceleration as following

![Nvidia_dla](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/dla.png)

This Nvidia interface can handle serving stage work(training will be done in cloud)

## HW Acceleration

1. Accelerator inside GPU:

![HW_accerlorator](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/HW_accerlorator.png)

Suitable for low end cellphone, because in house CPU and GPU is not powerful enough, liek 1 1080 UI will consume all resources, so need to separate HW Accelerator to do it

2. GPU in house

Most Android video, image can be done in CPU, and Android_NN supports CPU by default.

## Deep Learning Accelerator

![dl_accerlorator](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/dl_accerlorator.png)

Assuming 1T ops/s, so bandwidth is 5GB/s

1. if within CPU, we could use CPU ACP port. CPU set data Cacheable and Sharable

2. If between GPU, ACE port
