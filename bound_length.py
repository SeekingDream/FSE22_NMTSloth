import os
import time

import numpy as np
from tqdm import tqdm
import torch
import argparse

from utils import *

if not os.path.isdir('latency'):
    os.mkdir('latency')

RATE = 3

@torch.no_grad()
def measure_metric(adv_his):
    adv = adv_his[0][0]
    if type(adv) == list:
        adv = adv[0]
    ori_len = adv_his[0][1]
    max_length = len(adv.split(' ')) * RATE
    adv = [adv_his[i][1] for i in range(1, len(adv_his))]
    adv = [(min(a, max_length)/ori_len - 1) * 100 for a in adv]
    return np.array(adv).reshape([1, -1])


def main(data_id):
    data_name = MODEL_NAME_LIST[data_id]
    model, tokenizer, _ = load_model(data_name)
    final_inc = []
    for attack_id in [0, 1]:
        if attack_id == 0:
            attack_name = 'C'
        elif attack_id == 1:
            attack_name = 'W'
        else:
            attack_name = 'G'
        final_res_data = []
        adv_res = torch.load('adv/' + attack_name + '_' + data_name + '.adv')
        for adv in tqdm(adv_res):
            metric = measure_metric(adv)
            final_res_data.append(metric)
        inc = np.concatenate(final_res_data).mean(0)
        final_inc.append(inc.reshape([1, -1]))
    return final_inc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--data', default=2, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack type')
    args = parser.parse_args()
    save_res = []
    for data in [0, 1, 2]:
        final_res = main(data)
        save_res.extend(final_res)
    print()
    save_res = np.concatenate(save_res)
    np.savetxt('rate.csv', save_res, delimiter=',')
