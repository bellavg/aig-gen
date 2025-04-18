#!/bin/bash

 torchrun --standalone --nproc_per_node=4 train.py \
   configs/networks/small.py \
   configs/datasets/aig.py \
   configs/default.py
   
