import os
import time
from tqdm import tqdm
import torch
import argparse
import pyRAPL
import pynvml
from threading import Thread, Timer

from utils import *

if not os.path.isdir('latency'):
    os.mkdir('latency')

MAX_NUM = 100
MAX_PER = 3
# class CPUMeasurement(Thread):
#     def __init__(self):
#         super(CPUMeasurement, self).__init__()
#         self.energy = 0
#         pyRAPL.setup()
#         self.meter = pyRAPL.Measurement('bar')
#
#     def run(self):
#         while True:
#             self.meter.begin()
#             time.sleep(1)
#             self.meter.end()
#             new_e = sum(self.meter.result.dram + self.meter.result.pkg) / (10 ** 6)
#             if new_e is not None:
#                 if new_e > 0:
#                     self.energy += new_e
#
#     def end(self):
#         self.meter.end()
#         new_e = sum(self.meter.result.dram + self.meter.result.pkg) / (10 ** 6)
#         if new_e is not None:
#             if new_e > 0:
#                 self.energy += new_e
#         return self.energy


def handle_cpu_energy(a, b):
    if a is None and b is None:
        return -1
    else:
        if a is None:
            data = [d for d in b if d > 0]
        else:
            if b is None:
                data = [d for d in a if d > 0]
            else:
                data = [d for d in a + b if d > 0]
        return sum(data) / len(data) * 4


@torch.no_grad()
def measure_cpu(model, input_token, iter_num, name):
    cnt = 0
    while True:
        cnt += 1
        pyRAPL.setup()
        meter = pyRAPL.Measurement(name)
        meter.begin()
        pred = None
        t1 = time.time()
        for _ in range(iter_num):
            pred = model.generate(input_token)
        t2 = time.time()
        meter.end()
        latency = t2 - t1
        energy = handle_cpu_energy(meter.result.dram, meter.result.pkg)
        if energy != -1:
            return pred, latency, energy
        if cnt == 5:
            return pred, latency, -1


@torch.no_grad()
def measure_gpu(model, input_token, iter_num, device):
    pynvml.nvmlInit()
    device_id = 0 if device.index is None else device.index
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    t1 = time.time()
    power_list = []
    for _ in range(iter_num):
        model.generate(input_token)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_list.append(power)
    t2 = time.time()
    latency = t2 - t1
    s_energy = sum(power_list) / len(power_list) * latency
    energy = s_energy / (10 ** 6)
    pynvml.nvmlShutdown()
    return latency, energy


@torch.no_grad()
def measure_metric(model, tokenizer, adv_his, task_name):
    metric = {'flops': [], 'cpu': [], 'cuda': [], 'input_len': [], 'overheads': []}
    device_list = ['cpu', 'cuda']
    original_flops = []

    for device_name in device_list:
        device = torch.device(device_name)
        iter_num = 1 if device_name == 'cpu' else 3
        model = model.to(device)
        prev_adv = None
        for i, (adv, orig_len, overheads) in enumerate(adv_his):
            if device_name == 'cpu':
                original_flops.append(orig_len)
            if prev_adv == adv:
                if device_name == 'cpu':
                    metric['flops'].append(metric['flops'][-1])
                    metric['overheads'].append(overheads)
                metric[device_name].append(metric[device_name][-1])
                continue
            prev_adv = adv
            input_token = tokenizer(adv, return_tensors="pt", padding=True).input_ids
            input_token = input_token.to(device)
            name = task_name + '_' + str(i)
            if device_name == 'cpu':
                pred, latency, energy = measure_cpu(model, input_token, iter_num, name)
                metric['flops'].append(len(pred[0]))
                metric['overheads'].append(overheads)
                metric[device_name].append((latency / iter_num, energy / iter_num))
            else:
                latency, energy = measure_gpu(model, input_token, iter_num, device)
                metric[device_name].append((latency / iter_num, energy / iter_num))
    # new_flops = [min(d, 100) - 1 for d in metric['flops']]
    # assert new_flops == original_flops
    return metric, original_flops


def main(data_id, attack_id):
    data_name = MODEL_NAME_LIST[data_id]
    beam_size = BEAM_LIST[data_id]
    model, tokenizer, _, _, _ = load_model(data_name)
    task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(data_id)

    save_path = os.path.join('latency', task_name + '.latency')
    final_res_data = []
    adv_res = torch.load('adv/' + task_name + '_' + str(beam_size) + '.adv')
    if model.config.max_length < 100:
        model.config.max_length = 200
    for i, adv in tqdm(enumerate(adv_res)):
        if i >= MAX_NUM:
            return
        try:
            adv = adv[:MAX_PER + 1]
            metric = measure_metric(model, tokenizer, adv, task_name)
            final_res_data.append(metric)
            torch.save(final_res_data, save_path)
        except:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--data', default=1, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack type')
    args = parser.parse_args()
    main(args.data, args.attack)
    exit(0)
