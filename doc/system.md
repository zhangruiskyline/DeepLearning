<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [MPI](#mpi)
  - [MPI Reduce and Allreduce](#mpi-reduce-and-allreduce)
- [RABIT: A Reliable Allreduce and Broadcast Interface](#rabit-a-reliable-allreduce-and-broadcast-interface)
  - [limitation of map-reduce](#limitation-of-map-reduce)
    - [persistent resources requirements](#persistent-resources-requirements)
    - [All reduce](#all-reduce)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# MPI

http://mpitutorial.com

## MPI Reduce and Allreduce
http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/

![MPI_reduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/mpi_reduce.png)

![MPI_allreduce](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/mpi_allreduce.png)

# RABIT: A Reliable Allreduce and Broadcast Interface

Allreduce is an abstraction commonly used for solving machine learning problems. It is an operation where every node starts with a local value and ends up with an aggre- gate global result.

## limitation of map-reduce

### persistent resources requirements

machine learning programs usually consume more resources, they often demand allocation of tempo- ral results and caching. Since a machine learning process is generally iterative, it is desirable to make it __persistent__ across iterations.

### All reduce
An abstraction that overcomes such limitations is Allre- duce. Allreduce avoids the unnecessary map phases, real- location of memory and disk reads-writes between iterations.

* A single map phase takes place, followed by one or more Allre- duce stages
