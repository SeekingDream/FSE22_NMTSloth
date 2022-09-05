#!/bin/bash


#CUDA_VISIBLE_DEVICES=2 python measure_latency.py --data=1 --attack=0
#
#CUDA_VISIBLE_DEVICES=2 python measure_latency.py --data=1 --attack=2
#CUDA_VISIBLE_DEVICES=2 python measure_latency.py --data=1 --attack=3


CUDA_VISIBLE_DEVICES=2 python measure_senstive.py --data=1 --attack=0