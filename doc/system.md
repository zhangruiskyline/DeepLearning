<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

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

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

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
