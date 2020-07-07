<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Pytorch Framework](#pytorch-framework)
- [Pytorch Dataloader](#pytorch-dataloader)
  - [Multi-worker dataloader](#multi-worker-dataloader)
- [Pytorch Training accelerate](#pytorch-training-accelerate)
  - [General Rule](#general-rule)
  - [Distributed SGD](#distributed-sgd)
    - [Locking Updates (Synchronous SGD)](#locking-updates-synchronous-sgd)
    - [Async lock free SGD: Hogwild](#async-lock-free-sgd-hogwild)
    - [Pytorch implementation of Hogwild](#pytorch-implementation-of-hogwild)
      - [Leverage multi-processing](#leverage-multi-processing)
      - [Data Management](#data-management)
      - [Updates](#updates)
      - [Sample code](#sample-code)
    - [Parameter Server](#parameter-server)
  - [Data Parallel (DP)](#data-parallel-dp)
    - [DataParallelModel/DataParallelCriterion](#dataparallelmodeldataparallelcriterion)
  - [Distributed Data Parallel(DDP)](#distributed-data-parallelddp)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Pytorch Framework
# Pytorch Dataloader

## Multi-worker dataloader

# Pytorch Training accelerate

## General Rule

* Data loader speed up
* more CPU and networking utilization


## Distributed SGD 

https://towardsdatascience.com/accelerating-deep-learning-using-distributed-sgd-an-overview-e66c4aee1a0c

Stochastic Gradient Descent (SGD) and its multiple variants (such as RMSProp or Adam) are the crutial for deep learning training. In training, the mini-batches is strongly limited by computer memory (GPU memory if the algorithms run on GPUs). Due to these reasons, we need a fast and stable solution for parallelizing training over multiple independent nodes (computers) to achieve higher speedups.

Over the years, multiple solutions to this problem have been invented. In this post I will detail some of the most promising modifications to the vanilla SGD and try to explain the rationale behind these.

### Locking Updates (Synchronous SGD)


The simplest solution for parallelizing SGD across multiple CPUs or nodes is to have every node read the current parameter values, calculate the gradients using one batch of data, lock the parameters and update them. In pseudocode:

```
parallel for (batch of data):
   Acquire lock on current parameters
   Read current parameters
   Calculate the gradient w.r.t. the batch and update parameters
   Release lock
end
```

The problem with this kind of update is that acquiring the lock typically takes much longer than calculating the gradients. Thus, the synchronization becomes a major bottleneck and the parallelization is not effective. Therefore, alternatives have to be devised.

### Async lock free SGD: Hogwild

The workings of Hogwild ![Hogwild](http://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf) can be simply explained by: it is like the code above, just without the locks. In pseudocode:

```
parallel for (batch or sample of data):
   Read current parameters
   Calculate the gradient w.r.t. the batch and update parameters
end
```

Although this might not obvious, the authors of Hogwild were able to prove mathematically that it makes sense, at least in some learning settings. The main assumption of the algorithm is that the parameter updates are __sparse__. This means that most updates only update a sparse set of parameters. In this case, Hogwild is able to achieve almost the same convergence rate as the serial version.

### Pytorch implementation of Hogwild

https://towardsdatascience.com/this-is-hogwild-7cc80cd9b944

#### Leverage multi-processing

One possible algorithm is Hogwild! which leverages parallel processes on computers to run SGD simultaneously and asynchronously. Asynchronicity is important for reducing unnecessary idle time in which no calculations are done but energy is still consumed.

__torch.multiprocessing__ is a drop in replacement for Python’s multiprocessing module. It supports the exact same operations, but extends it, so that all tensors sent through a multiprocessing.Queue, will have their data moved into __shared memory__ and will only send a handle to another process.



#### Data Management
Using Hogwild!, each participating process in a parallel training session is responsible for one partition of the data, e.g. having 8 parallel processes would split the data set in 8 equal parts whereas each process is assigned to one part. Moreover, on shared memory or a separate server, an initial model is created which can be accessed by all processes.


#### Updates
Once the training starts, each process loads the current state of the model from __shared memory__ and starts reading the first batch of their data partition. As in standard SGD, each process is calculating the gradients for that batch. The gradients are now directly written to the shared model without blocking the other processes. Once written, the new model parameters are loaded and the next batch is used. Due to the missing blocking, the shared model will sometimes receive old gradients which, one may think, could be a disadvantage. The researchers of Hogwild!, however, could show that the training even benefits from the non-blocking fashion.


#### Sample code

```python
import torch.multiprocessing as mp
from model import MyModel

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```

### Parameter Server

![Parameter Server](https://github.com/zhangruiskyline/DeepLearning/blob/master/img/PS.png)

was invented by researchers in Google as part of their DistBelief framework (predecessor of TensorFlow) and its variants are still used in today’s distributed TensorFlow framework. It makes use of multiple model replicas that each calculate updates based on a small portion of data (data parallelism). After calculation, the updates are sent to a centralized parameter server. The parameter server itself is split into multiple nodes, each holding and updating a small subset of the parameters (model parallelism). This parallelization scheme is very popular up to this date, not least due to it being a built-in feature in TensorFlow. However, convergence of this scheme suffers from the fact that model replicas do not share parameters (weights) or updates with each other. This means that their parameters can always diverge.


## Data Parallel (DP)

https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2

General rule is to distribute the data into different sub data sets, and each GPU will process a subset of them. As shown in 

![DP](https://github.com/zhangruiskyline/DeepLearning/blob/master/img/DP.png)

The process steps will be like

* The mini-batch is split on GPU:0
* Split and move min-batch to all different GPUs
* Copy model out to GPUs
* Forward pass occurs in all different GPUs
* Compute loss with regards to the network outputs on GPU:0, and return losses to the different GPUs. Calculate gradients on each GPU
* Sum up gradients on GPU:0 and use the optimizer to update model on GPU:0


```Python
parallel_model = torch.nn.DataParallel(model) # Encapsulate the model

predictions = parallel_model(inputs)          # Forward pass on multi-GPUs
loss = loss_function(predictions, labels)     # Compute loss function on single GPU
loss.mean().backward()                        # Average GPU-losses + backward pass
optimizer.step()                              # Optimizer step
predictions = parallel_model(inputs)          # Forward pass with new parameters
```

### DataParallelModel/DataParallelCriterion

There are two main solution to the imbalanced GPU usage issue:

* computing the loss in the forward pass of your model,
* computing the loss in a parallel fashion.

We will focus on solution 2, the solution is to keep each partial output on its GPU instead of gathering all of them to GPU-1. We well need to distribute our loss criterion computation as well to be able to compute and back propagate our loss.

we can check PyTorch package called PyTorch-Encoding(https://github.com/zhanghang1989/PyTorch-Encoding) which comprises these custom parallelization functions.



If you use __DataParallelCriterion__, you should not wrap your model with a regular _DataParallel_. This is because _DataParallel_ basically gathers output on one GPU. Therefore, we use the __DataParallelModel__, a Custom DataParallel class. The process of learning using DataParallelModel and DataParallelCriterion is as follows: How to use is quite simple. Just import the parallel.py file from the Pytorch-Encoding package and make it import in the training code.

The new way can be illustrated as

![DP2](https://github.com/zhangruiskyline/DeepLearning/blob/master/img/DP2.png)

And the code sample can be like

```python
from parallel import DataParallelModel, DataParallelCriterion

parallel_model = DataParallelModel(model)             # Encapsulate the model
parallel_loss  = DataParallelCriterion(loss_function) # Encapsulate the loss function

predictions = parallel_model(inputs)      # Parallel forward pass
                                          # "predictions" is a tuple of n_gpu tensors
loss = parallel_loss(predictions, labels) # Compute loss function in parallel
loss.backward()                           # Backward pass
optimizer.step()                          # Optimizer step
predictions = parallel_model(inputs)      # Parallel forward pass with new parameters
```

The difference between _DataParallelModel_ and _torch.nn.DataParallel_ is just that the output of the forward pass (predictions) is not gathered on GPU-1 and is thus a tuple of n_gpu tensors, each tensor being located on a respective GPU.

The _DataParallelCriterion_ container encapsulate the loss function and takes as input the tuple of n_gpu tensors and the target labels tensor. It computes the loss function in parallel on each GPU, splitting the target label tensor the same way the model input was chunked by _DataParallel_.


## Distributed Data Parallel(DDP)

https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

* Multiprocessing with __DistributedDataParallel__ duplicates the model across multiple GPUs, each of which is controlled by one process. 

* The GPUs can all be on the same node or spread across multiple nodes. 

* Every process does __identical__ tasks, and each process communicates with all the others. Only __gradients__ are passed between the processes/GPUs so that network communication is less of a bottleneck.


> Training process 

1. each process loads its own minibatches from disk and passes them to its GPU

2. Each GPU does its own forward pass, and then the gradients are __all-reduced__ across the GPUs. Gradients for each layer do not depend on previous layers, so the gradient all-reduce is calculated concurrently with the backwards pass to futher alleviate the networking bottleneck

3. At the end of the backwards pass, every node has the __averaged__ gradients, ensuring that the model weights stay __synchronized__.

DDP  can be illustrated as

![DDP](https://github.com/zhangruiskyline/DeepLearning/blob/master/img/DDP.png)

> Code requirement

All this requires that the multiple processes, possibly on multiple nodes, are synchronized and communicate. Pytorch does this through its ```distributed.init_process_group``` function. This function needs to know where to find process 0 so that all the processes can sync up and the total number of processes to expect. Each individual process also needs to know the total number of processes as well as its rank within the processes and which GPU to use. It’s common to call the total number of processes the world size. Finally, each process needs to know which slice of the data to work on so that the batches are non-overlapping. Pytorch provides ```nn.utils.data.DistributedSampler``` to accomplish this.




