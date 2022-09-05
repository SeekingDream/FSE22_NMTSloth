#!/bin/bash


#CUDA_VISIBLE_DEVICES=3 python measure_latency.py --data=2 --attack=0
#
#
#CUDA_VISIBLE_DEVICES=3 python measure_latency.py --data=2 --attack=2
#CUDA_VISIBLE_DEVICES=3 python measure_latency.py --data=2 --attack=3


CUDA_VISIBLE_DEVICES=3 python measure_senstive.py --data=2 --attack=0