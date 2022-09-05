import os
import torch
import numpy as np

from utils import MODEL_NAME_LIST
latency_dir = 'latency'
save_dir = 'res/severity'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
MAX_PER = 3

for task_id in range(4):
    model_name = MODEL_NAME_LIST[task_id]
    subj_res = []
    for attack_id in [6]:
        task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(task_id)
        latency_file = os.path.join(latency_dir, task_name + '.latency')
        res_list = torch.load(latency_file)
        i_flops, i_latency_cpu, i_energy_cpu, i_latency_gpu, i_energy_gpu = [], [], [], [], []
        overhead_list = []

        cpu_overheads = np.concatenate([np.array(res['cpu'])[0].reshape([1, -1]) for (res, _) in res_list])
        print(task_id, cpu_overheads.mean(0))
        for (res, _) in res_list:
            flops = np.array(res['flops'])
            cpu = np.array(res['cpu'])
            gpu = np.array(res['cuda'])

            cpu_i = (cpu / cpu[0] - 1) * 100
            gpu_i = (gpu / gpu[0] - 1) * 100
            flops_i = (flops / flops[0] - 1) * 100
            i_flops.append(flops_i.reshape([1, -1]))
            i_latency_cpu.append(cpu_i[:, 0].reshape([1, -1]))
            i_energy_cpu.append(cpu_i[:, 1].reshape([1, -1]))
            i_latency_gpu.append(gpu_i[:, 0].reshape([1, -1]))
            i_energy_gpu.append(gpu_i[:, 1].reshape([1, -1]))
        i_flops = np.concatenate(i_flops)
        i_latency_cpu = np.concatenate(i_latency_cpu)
        i_energy_cpu = np.concatenate(i_energy_cpu)
        i_latency_gpu = np.concatenate(i_latency_gpu)
        i_energy_gpu = np.concatenate(i_energy_gpu)

        # print(
        #     len(res_list), attack_id,
        #     i_flops.mean(0), '\n',
        #     i_latency_cpu.mean(0), '\n',
        #     i_energy_cpu.mean(0), '\n',
        #     i_latency_gpu.mean(0),  '\n',
        #     i_energy_gpu.mean(0)
        # )
        tmp_res = [
            i_flops.mean(0)[1:MAX_PER + 1],
            i_latency_cpu.mean(0)[1:MAX_PER + 1],
            i_energy_cpu.mean(0)[1:MAX_PER + 1],
            i_latency_gpu.mean(0)[1:MAX_PER + 1],
            i_energy_gpu.mean(0)[1:MAX_PER + 1]
        ]
        tmp_res = np.concatenate(tmp_res).reshape([1, -1])
        subj_res.append(tmp_res)
        print('-----------------')
    subj_res = np.concatenate(subj_res)
    np.savetxt(os.path.join(save_dir, 'new_' + str(task_id) + '.csv'), subj_res, delimiter=',')
