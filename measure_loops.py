import os
import torch
import numpy as np

from utils import MODEL_NAME_LIST

max_length = 200

sel_per = 3

if not os.path.isdir('loop_res'):
    os.mkdir('loop_res')


for attack_name in ['C', 'W']:
    for method in ['latency', 'baseline']:
        final_table = []
        for data_name in MODEL_NAME_LIST:
            latency_file = os.path.join('latency', data_name + '_' + attack_name + '_' + str(max_length) + '.' + method)
            dist_data_path = os.path.join('loop_res', data_name + '_' + attack_name + '_' + method + '.csv')
            latency_res = torch.load(latency_file)
            I_flops, I_cpu, I_gpu = [], [], []
            flops, cpu, gpu = [], [], []

            for data in latency_res:
                I_flops.append(((np.array(data['flops'][1:]) + 1) / (data['flops'][0] + 1) - 1).reshape([1, -1]))
                I_cpu.append((np.array(data['cpu'][1:]) / data['cpu'][0] - 1).reshape([1, -1]))
                I_gpu.append((np.array(data['cuda'][1:]) / data['cuda'][0] - 1).reshape([1, -1]))

                flops.append((np.array(data['flops']) + 1)[[0, sel_per]].reshape([1, -1]))
                cpu.append(np.array(data['cpu'])[[0, sel_per]].reshape([1, -1]))
                gpu.append(np.array(data['cuda'])[[0, sel_per]].reshape([1, -1]))

            I_flops, I_cpu, I_gpu = np.concatenate(I_flops), np.concatenate(I_cpu), np.concatenate(I_gpu)
            I_flops_avg, I_flops_max = I_flops.mean(axis=0).reshape([-1, 1]), I_flops.max(axis=0).reshape([-1, 1])
            I_cpu_avg, I_cpu_max = I_cpu.mean(axis=0).reshape([-1, 1]), I_cpu.max(axis=0).reshape([-1, 1])
            I_gpu_avg, I_gpu_max = I_gpu.mean(axis=0).reshape([-1, 1]), I_gpu.max(axis=0).reshape([-1, 1])
            data_res = np.concatenate([I_flops_avg, I_flops_max, I_cpu_avg, I_cpu_max, I_gpu_avg, I_gpu_max], axis=1)
            final_table.append(data_res)

            flops = np.concatenate(flops)
            cpu = np.concatenate(cpu)
            gpu = np.concatenate(gpu)
            tmp_res = np.concatenate([flops, cpu, gpu], axis=1)
            tmp_res = np.around(tmp_res, decimals=2)
            np.savetxt(dist_data_path, tmp_res, delimiter=',')

        final_table = np.concatenate(final_table)
        final_table = final_table * 100
        save_path = os.path.join('loop_res', method + '_' + attack_name + '.csv')
        final_table = np.around(final_table, decimals=2)
        np.savetxt(save_path, final_table, delimiter=',')