#!/bin/bash

#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=3 --attack=0
#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=3 --attack=2


#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=2 --attack=0 --beam=1
#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=2 --attack=0 --beam=2
#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=2 --attack=0 --beam=3
#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=2 --attack=0 --beam=4
#CUDA_VISIBLE_DEVICES=2 python generate_adv.py --data=2 --attack=0 --beam=5


CUDA_VISIBLE_DEVICES=3 python measure_senstive.py --data=3 --attack=0