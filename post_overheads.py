import os
import torch
import numpy as np

from utils import MODEL_NAME_LIST, BEAM_LIST
adv_dir = 'adv'
save_dir = 'res/overhead'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
MAX_PER = 3

save_res = []
for task_id in range(4):
    model_name = MODEL_NAME_LIST[task_id]
    overheads_res = []
    beam_size = BEAM_LIST[task_id]
    for attack_id in [0, 1, 6]:
        task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(task_id) + '_' + str(beam_size)
        adv_file = os.path.join(adv_dir, task_name + '.adv')
        adv_res = torch.load(adv_file)
        overheads = [np.array([d[p][-1] for d in adv_res]).reshape([-1, 1]) for p in range(1, MAX_PER + 1)]
        overheads = np.concatenate(overheads, axis=1)
        overheads_res.append(overheads)
    new_data = np.concatenate(overheads_res, axis=1)
    new_data = new_data.mean(0).reshape([1, -1])
    save_res.append(new_data)
save_res = np.concatenate(save_res)
np.savetxt(
    os.path.join(save_dir, 'final.csv'),
    save_res, delimiter=','
)



