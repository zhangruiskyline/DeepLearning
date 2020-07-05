<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Pytorch Framework](#pytorch-framework)
- [Pytorch Dataloader](#pytorch-dataloader)
  - [Multi-worker dataloader](#multi-worker-dataloader)
- [Pytorch Training accelerate](#pytorch-training-accelerate)
  - [General Rule](#general-rule)
  - [Hogwild](#hogwild)
  - [Data Parallel (DP)](#data-parallel-dp)
  - [Distributed Data Parallel(DDP)](#distributed-data-parallelddp)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Pytorch Framework
# Pytorch Dataloader

## Multi-worker dataloader

# Pytorch Training accelerate

## General Rule

* Data loader speed up
* more CPU and networking utilization


## Hogwild

## Data Parallel (DP)

https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2

General rule is to distribute the data into different sub data sets, and each GPU will process a subset of them. As shown in 

![dp](https://github.com/zhangruiskyline/DeepLearning/blob/master/img/dp.png)

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
loss = loss_function(predictions, labels)     # Compute loss function
loss.mean().backward()                        # Average GPU-losses + backward pass
optimizer.step()                              # Optimizer step
predictions = parallel_model(inputs)          # Forward pass with new parameters
```

## Distributed Data Parallel(DDP)
