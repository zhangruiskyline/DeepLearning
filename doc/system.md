<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Distributed Machine Learning](#distributed-machine-learning)
  - [model parallelism vs data parallelism](#model-parallelism-vs-data-parallelism)
  - [Data Parallelism](#data-parallelism)
    - [Parameter Averaging](#parameter-averaging)
    - [Update based method](#update-based-method)
    - [Asychronous Stochastic Gradient Descent](#asychronous-stochastic-gradient-descent)
  - [Distributed Asychronous Stochastic Gradient Descent](#distributed-asychronous-stochastic-gradient-descent)
- [Parameter Server](#parameter-server)
  - [System view](#system-view)
    - [Challenge](#challenge)
    - [Large data](#large-data)
    - [Synchronization](#synchronization)
    - [Fault tolerance](#fault-tolerance)
- [MPI](#mpi)
  - [MPI Reduce and Allreduce](#mpi-reduce-and-allreduce)
- [RABIT: A Reliable Allreduce and Broadcast Interface](#rabit-a-reliable-allreduce-and-broadcast-interface)
  - [limitation of map-reduce](#limitation-of-map-reduce)
    - [persistent resources requirements](#persistent-resources-requirements)
  - [All reduce](#all-reduce)
    - [allreduce in practice](#allreduce-in-practice)
  - [Rabit Algorithm](#rabit-algorithm)
    - [Fault-tolerant](#fault-tolerant)
      - [Recovery](#recovery)
      - [Consensus Protocol](#consensus-protocol)
      - [Routing](#routing)
- [Tree based Allreduce vs Ring based Allreduce](#tree-based-allreduce-vs-ring-based-allreduce)
  - [limitations of Tree Allreduce](#limitations-of-tree-allreduce)
  - [Ring Allreduce](#ring-allreduce)
    - [Scatter-Reduce](#scatter-reduce)
    - [AllGather](#allgather)
    - [Overhead Analysis](#overhead-analysis)
- [Allreduce vs Paramter server](#allreduce-vs-paramter-server)
- [Nvidia Multi-GPU Lib: NCCL](#nvidia-multi-gpu-lib-nccl)
  - [communication primitive](#communication-primitive)
  - [Ring-based Collective communication](#ring-based-collective-communication)
  - [NCCL Implementation](#nccl-implementation)
    - [Performance](#performance)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Distributed Machine Learning

http://dlsys.cs.washington.edu/schedule

## model parallelism vs data parallelism

In model parallelism, different machines in the distributed system are responsible for the computations in different parts of a single network - for example, each layer in the neural network may be assigned to a different machine.

In data parallelism, different machines have a complete copy of the model; each machine simply gets a different portion of the data, and results from each are somehow combined.

Of course, these approaches are not mutually exclusive. Consider a cluster of multi-GPU systems. We could use model parallelism (model split across GPUs) for each machine, and data parallelism between machines.

## Data Parallelism

Data parallel approaches to distributed training keep a copy of the entire model on each worker machine, processing different subsets of the training data set on each. Data parallel training approaches all require some method of combining results and synchronizing the model parameters between each worker. A number of different approaches have been discussed in the literature, and the primary differences between approaches are

* Parameter averaging vs. update (gradient)-based approaches
* Synchronous vs. asynchronous methods
* Centralized vs. distributed synchronization

### Parameter Averaging
Parameter averaging is the conceptually simplest approach to data parallelism. With parameter averaging, training proceeds as follows:

1. Initialize the network parameters randomly based on the model configuration
2. Distribute a copy of the current parameters to each worker
3. Train each worker on a subset of the data
4. Set the global parameters to the average the parameters from each worker
5. While there is more data to process, go to step 2

![parameter_average](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/parameter_average.png)

Steps 2 through 4 are demonstrated in the image below. In this diagram, W represents the parameters (weights, biases) in the neural network. Subscripts are used to index the version of the parameters over time, and where necessary for each worker machine.

we instead perform learning on m examples in each of the n workers (where worker 1 gets examples 1, ..., m, worker 2 gets examples m + 1, ..., 2m and so on), we have:

![parameter_average_math](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/parameter_average_math.png)


Now, parameter averaging is conceptually simple, but there are a few complications that we’ve glossed over.

First, how should we implement averaging? The naive approach is to simply average the parameters after each iteration. While this can work, we are likely to find that the overhead of doing so to be impractically high; network communication and synchronization costs may overwhelm the benefit obtained from the extra machines. Consequently, parameter averaging is generally implemented with an averaging period (in terms of number of minibatches per worker) greater than 1. However, if we average too infrequently, the local parameters in each worker may diverge too much, resulting in a poor model after averaging. The intuition here is that the average of N different local minima are not guaranteed to be a local minima:

### Update based method

A conceptually similar approach to parameter averaging is what we might call ‘update based’ data parallelism. The primary difference between the two is that instead of transferring parameters from the workers to the parameter server, we will transfer the updates (i.e., gradients post learning rate and momentum, etc.) instead. This gives an update of the form:

![Stochastic_average_math](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/stochastic_average_math.png)

The worker machine diagram looks like:

![Stochastic_average](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/stochastic_average.png)

If we again define our loss function as L, then parameter vector W at iteration i + 1 for simple SGD training with learning rate α is obtained by:

![Stochastic_average_math_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/Stochastic_average_math_2.png)

![Stochastic_average_math_3](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/Stochastic_average_math_3.png)

Consequently, there is an equivalence between parameter averaging and update-based data parallelism, when parameters are updated synchronously (this last part is key). This equivalence also holds for multiple averaging steps and other updaters (not just simple SGD).

### Asychronous Stochastic Gradient Descent

Update-based data parallelism becomes more interesting (and arguably more useful) when we relax the synchronous update requirement. That is, by allowing the updates ∆Wi,j to be applied to the parameter vector as soon as they are computed (instead of waiting for N ≥ 1 iterations by all workers), we obtain asynchronous stochastic gradient descent algorithm. Async SGD has two main benefits:

* First, we can potentially gain higher throughput in our distributed system: workers can spend more time performing useful computations, instead of waiting around for the parameter averaging step to be completed.

* Second, workers can potentially incorporate information (parameter updates) from other workers sooner than when using synchronous (every N steps) updating.

By introducing asynchronous updates to the parameter vector, we introduce a new problem, known as the stale gradient problem. The stale gradient problem is quite simple: the calculation of gradients (updates) takes time. By the time a worker has finished these calculations and applies the results to the global parameter vector, the parameters may have been updated a number of times. This problem is illustrated in the figure below.

![stale_gradient](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/stale_gradient.png)

Most variants of asynchronous stochastic gradient descent maintain the same basic approach, but apply a variety of strategies to minimize the impact of the stale gradients, whilst attempting to maintaining high cluster utilization. It should be noted that parameter averaging is not subject to the stale gradient problem due to the synchronous nature of the algorithm.

* Scaling the value λ separately for each update ∆Wi,j based on the staleness of the gradients

* Implementing ‘soft’ synchronization protocols ([9])

* Use synchronization to bound staleness. For example, the system of [4] delays faster workers when necessary, to ensure that the maximum staleness is below some threshold

## Distributed Asychronous Stochastic Gradient Descent

* No centralized parameter server is present in the system (instead, peer to peer communication is used to transmit model updates between workers).
* Updates are heavily compressed, resulting in the size of network communications being reduced by some 3 orders of magnitude.

# Parameter Server

## System view
* Initialize model with small random values Paid Once
  * fairly trivial to parallelize
* Try to train the right answer for your input set Iterate through the input set many many times
* Adjust the model: Send a small update to the model parameters

### Challenge

* Three main challenges of implementing a parameter server:
  * Accessing parameters requires lots of network bandwidth
  * Training is sequential and synchronization is hard to scale
  * Fault tolerance at scale (~25% failure rate for 10k machine-hour jobs)

* General purpose machine-learning frameworks
  * Many have synchronization points -> difficult to scale
  * Key observation: cache state between iterations

### Large data

What are parameters of a ML model? Usually an element of a vector, matrix, etc. Need to do lots of linear algebra operations.

* Introduce new constraint: ordered keys
* Typically some index into a linear algebra structure
* High model complexity leads to overfitting: Updates don’t touch many parameters
  * Range push-and-pull: Can update a range of rows instead of single key
  * When sending ranges, use compression

### Synchronization

* ML models try to find a good local min/min
* Need updates to be generally in the right direction
* Not important to have strong consistency guarantees all the time
* Parameter server introduces Bounded Delay

### Fault tolerance
* Server stores all state, workers are stateless
  * However, workers cache state across iterations
* Keys are replicated for fault tolerance
* Jobs are rerun if a worker fails

# MPI

http://mpitutorial.com

## MPI Reduce and Allreduce
http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/

![MPI_reduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/mpi_reduce.png)

![MPI_allreduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/mpi_allreduce.png)

# RABIT: A Reliable Allreduce and Broadcast Interface
http://homes.cs.washington.edu/~icano/projects/rabit.pdf

Allreduce is an abstraction commonly used for solving machine learning problems. It is an operation where every node starts with a local value and ends up with an aggre- gate global result.

## limitation of map-reduce

### persistent resources requirements

machine learning programs usually consume more resources, they often demand allocation of tempo- ral results and caching. Since a machine learning process is generally iterative, it is desirable to make it __persistent__ across iterations.

## All reduce
An abstraction that overcomes such limitations is Allre- duce. Allreduce avoids the unnecessary map phases, real- location of memory and disk reads-writes between iterations.

* A single map phase takes place, followed by one or more Allreduce stages

* The program persists across iterations, and it re-uses the resources allocated previously. It is therefore a more convenient abstraction for solving many large scale machine learning problems.

![reduce_allreduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/reduce_allreduce.png)

* OpenMPI is non fault-tolerant

### allreduce in practice

In Allreduce settings, nodes are organized in a tree structure. Each node holds a portion of the data and computes some values on it. Those values are passed up the tree and aggregated, until a global aggregate value is calculated in the root node (reduce). The global value is then passed down to all other nodes (broadcast).

![tree_allreduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/tree_allreduce.png)

Several machine learning problems fit into this abstrac- tion, where every node works on some portion of the data, computes some local statistics (e.g. local gradients) and then invokes the Allreduce primitive to get a global ag- gregated result (e.g. global gradient) in order to further proceed with the computations.

## Rabit Algorithm

* Flexibility in programming: Programs can call Ra- bit functions in any order, as opposed to frameworks where callbacks are offered and called at specific points in time.
* Programs persistence: Programs continue running over all iterations, instead of being restarted on each one of them.
* Fault tolerance: Rabit programs can recover their state (e.g. machine learning model) using synchronous function calls.
* MPIcompatible: Code that uses Rabit API also compiles with existing MPI compilers, i.e. users can use MPI Allreduce with no code modification.

### Fault-tolerant

1. Pause every node until the failed node is fully recov- ered.
2. Detect the model version we need to recover by ob- taining the minimum operation number. This is done using the Consensus Protocol.
3. Transfer the model to the failed node using the Routing Protocol.
4. Resume the execution of the failed node using the received model.
5. Resume the execution of the other nodes as soon as the failed node catches up

![allreduce_recovery](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allreduce_recovery.png)

#### Recovery
Message Passing: nodes send messages to their neighbors to find out information about them. It can be used in many algorithms such as Shortest Path

#### Consensus Protocol
The consensus protocol agrees on the model version to recover using an Allreduce min operation. In the example shown in Figure,
![min_recovery](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/min_recovery.png)
the value of each node is the model version they store. The green node on the left subtree is recovering so it wants to find which model it needs in order to make progress. It does so with the Allreduce min operation. After the value 0 is broadcast-ed to every node, everyone knows that model 0 is the version to be used
for recovery.

#### Routing
Routing: Once the model version to recover is agreed among the nodes, the routing protocol executes. It finds the shortest path a failed node needs to use in order to retrieve the model. It uses two rounds of message passing.
* The first round computes the distance from a failed node to the nearest node that has the model.
* The second round sends requests through the shortest path until it reaches the node that owns the model, and retrieves it.

# Tree based Allreduce vs Ring based Allreduce

## limitations of Tree Allreduce
![allreduce_issue](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allreduce_issue.png)

If we have 300M parameters and each is 4 Bytes. it will have 1.2G data, and assuming bandwidth is 1GB/s. if we use 2 GPUs, delayed 1.2s, 10 GPUs, delayed 10.8s for each epoch

## Ring Allreduce
GPUs have been placed in logical ring. GPU get data from left GPU and sends to right GPU
![ring_reduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ring_reduce.png)

### Scatter-Reduce
assuming we have N GPUs in system and every GPU has same length array. GPU will chunk the data into N small pieces.

GPU will execute N-1 times iterations. In every iteration, sender and receiver are different. n GPU will send n block and receive n-1  

```
GPU #, Send, receive
0 Chunk 0 Chunk 4
1 Chunk 1 Chunk 0
2 Chunk 2 Chunk 1
3 Chunk 3 Chunk 2
4 Chunk 4 Chunk 3
```

The figure shows the data flow for one iteration
![ringreduce_data](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ringreduce_data.png)

This figure shows middle results after one iteration
![ringreduce_middle](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ringreduce_middle.png)

At the end, every GPU block will have all results from same blocks along other GPUs
![ringreduce_end](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/ringreduce_end.png)

### AllGather
After scatter reduce, every GPU has some values from all GPUs, they need to exchange these values so all GPUs have all values

allgather is same as scatter reduce, except that it is not sum, just simple copy, the process will look like
![allgather](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allgather.png)

And after N-1 iteration, final results will be
![allgather_end](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allgather_end.png)

### Overhead Analysis
* Communication cost

N GPUs, in Scatter-Reduce stage, each one will have N-1 data receiving, and send k data(K is array length in single GPU), in allgather stage, another N-1 receving and N sending

For Each GPU,
```
2(N−1)⋅K
```

# Allreduce vs Paramter server
http://hunch.net/?p=151364

# Nvidia Multi-GPU Lib: NCCL

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines, such as all-gather, reduce, broadcast, etc., that have been optimized to achieve high bandwidth over PCIe, Nvlink、InfiniBand

## communication primitive

* Reduce:

Multiple senders send information and combine in one node

![reduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/reduce.png)

* All reduce

Multiple senders send information and combine and distribute to all nodes

![all_reduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/all_reduce.png)

Traditional Collective communication assumes a tree style topology, but it dose not assume a flat tree, instead it use ring-based Collective communication.

## Ring-based Collective communication

Ring based assumes all nodes are in a circle.

* Assume there are K nodes, N data, and bandwidth is B

total communication time is:

```
(K-1)*N/B
```

* Split the data into S, then we split N/S data

total communication time is:

```
S*(N/S/B) + (k-2)* (N/S/B) = N(S+K-2)/(SB) --> N/B

if S>>K, which is normally true
```

![collective_communication](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/collective_communication.png)

* Implement in GPU environment

![GPU_allreduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/GPU_allreduce.png)

## NCCL Implementation

NCCL Implements CUDA C++ kernels. including 3 primitives: Copy，Reduce，ReduceAndCopy. NCCL can support PCIe、NVlink、GPU Direct P2P. NCCL 2.0 can support many machines using Sockets (Ethernet), InfiniBand with GPU Direct RDMA

![NCCL_mode](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/NCCL_mode.png)


### Performance

* Single node multiple GPU Cards

![allreduce_perf_1](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allreduce_perf_1.png)

DGX_1 can achieve 60G bps, the former 3 is single machine multiple GPU cards. first is 2 GPU connects via 2 CPUs with QPI, second is 2 GPUs connect with switch, and finally with CPU. third one is all four GPU cards via PCIe.

* Multiple nodes and Multiple GPUs

![allreduce_perf_2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/allreduce_perf_2.png)

PCIe within machine and InfiniBand among nodes
