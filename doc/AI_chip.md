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
- [Intermediate representation (IR)](#intermediate-representation-ir)
  - [Why need IR](#why-need-ir)
  - [XLA: Accelerated Linear Algebra](#xla-accelerated-linear-algebra)
    - [Architecture](#architecture)
  - [TVM: An End to End IR Stack for Deploying the Deep Learning Workloads to Hardwares](#tvm-an-end-to-end-ir-stack-for-deploying-the-deep-learning-workloads-to-hardwares)

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


# Intermediate representation (IR)

referring https://mp.weixin.qq.com/s/0iDVjaucRUpn2UrVBuQ-oQ

An Intermediate representation (IR) is the data structure or code used internally by a compiler or virtual machine to represent source code. An IR is designed to be conducive for further processing, such as optimization and translation. A "good" IR must be accurate – capable of representing the source code without loss of information – and independent of any particular source or target language. An IR may take one of several forms: an in-memory data structure, or a special tuple- or stack-based code readable by the program. In the latter case it is also called an intermediate language.

## Why need IR

* Bridge the Frontend different frameworks and backend different HW

![AI_IR](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/AI_IR.png)

This is a new interesting era of deep learning, with emergence trend of new system, hardware and computational model. The use case for deep learning is more heterogeneous, and we need tailored learning system for our cars, mobiles and cloud services. The future of deep learning system is going to be more heterogeneous, and we will find emergence need of different front-ends, backends and optimization techniques. Instead of building a monolithic solution to solve all these problems, how about adopt unix philosophy, build effective modules for learning system, and assemble them together to build minimum and effective systems

## XLA: Accelerated Linear Algebra

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that optimizes TensorFlow computations. The results are improvements in speed, memory usage, and portability on server and mobile platforms. Initially, most users will not see large benefits from XLA, but are welcome to experiment by using XLA via just-in-time (JIT) compilation or ahead-of-time (AOT) compilation. Developers targeting new hardware accelerators are especially encouraged to try out XLA.


1. Existing CPU architecture not yet officially supported by XLA, with or without an existing LLVM backend.
2. Non-CPU-like hardware with an existing LLVM backend.
3. Non-CPU-like hardware without an existing LLVM backend.

* XLA is only for TensorFlow

### Architecture

The input language to XLA is called "HLO IR", or just HLO (High Level Optimizer). The semantics of HLO are described on the Operation Semantics page. It is most convenient to think of HLO as a compiler IR.

## TVM: An End to End IR Stack for Deploying the Deep Learning Workloads to Hardwares

![TVM](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/TVM.png)

A lot of powerful optimizations can be supported by the graph optimization framework. However we find that the computational graph based IR (NNVM) alone is not enough to solve the challenge of supporting different hardware backends

![TVM_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/TVM_2.png)
