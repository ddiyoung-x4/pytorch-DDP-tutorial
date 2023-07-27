# pytorch-DDP-tutorial
This is an example implementation of Pytorch DistributedDataParallel(DDP).


**How to Run**
```
torchrun --nproc_per_node=your_num_of_multi-process train.py --gpu_ids your_gpu_ids
ex) below
torchrun --nproc_per_node=2 train.py --gpu_ids 0,1
```
