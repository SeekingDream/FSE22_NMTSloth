import os
import torch
import numpy as np

from utils import MODEL_NAME_LIST
latency_dir = 'senstive'
save_dir = 'res/senstive'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
MAX_PER = 3


for model_id in range(4):
    save_res = {'beam': []}
    for attack_id in [0, 1, 6]:
        model_name = MODEL_NAME_LIST[model_id]
        task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(model_id)
        final_res = torch.load('senstive/' + task_name + '.sen')
        for key in save_res:
            key_res = []
            for index in range(len(final_res[key])):
                data = final_res[key][index]
                inc = data[:, 1:] / data[:, 0:1]
                inc[:, 1] = inc[:, :2].max(1)
                inc[:, 2] = inc[:, :3].max(1)
                inc = (inc.mean(0) - 1) * 100
                key_res.append(inc.reshape([1, -1]))
            key_res = np.concatenate(key_res)
            save_res[key].append(key_res)
    for key in save_res:
        save_name = os.path.join(save_dir, str(model_id) + '_' + key + '.csv')
        save_res[key] = np.concatenate(save_res[key])
        np.savetxt(save_name, save_res[key], delimiter=',')




