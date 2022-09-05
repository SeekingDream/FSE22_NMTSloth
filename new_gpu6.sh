#!/bin/bash


#
#CUDA_VISIBLE_DEVICES=6 python measure_latency.py --data=2 --attack=1
#
#
#CUDA_VISIBLE_DEVICES=6 python measure_latency.py --data=2 --attack=4
#CUDA_VISIBLE_DEVICES=6 python measure_latency.py --data=2 --attack=5


CUDA_VISIBLE_DEVICES=6 python measure_senstive.py --data=2 --attack=1