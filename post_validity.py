import os
import torch
import numpy as np
from tqdm import tqdm

from utils import MODEL_NAME_LIST, load_model_dataset

senstive_dir = 'senstive'
save_dir = 'res/validity'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
MAX_PER = 3
device = torch.device(7)


def main():
    var_dict = torch.load("preliminary/MultiUN.var")
    var_res = var_dict['de-en']
    for beam_id in range(5):
        for task_id in range(4):
            final_res = []
            for attack_id in [0, 1, 6]:
                task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(task_id)
                save_path = os.path.join(senstive_dir, task_name + '.sen')
                sen_res = torch.load(save_path)
                beam_loops = sen_res['beam'][beam_id]
                var = [var_res[d[0]] for d in beam_loops]
                var = np.sqrt(var)
                var = np.tile(np.array(var).reshape([-1, 1]), [1, 3])

                rate = (beam_loops[:, 1:] - beam_loops[:, 0:1]) / var
                res = [(rate > i).sum(0).reshape([1, -1]) for i in range(11)]
                res = np.concatenate(res)
                res = res / len(var) * 100
                final_res.append(res[:, 2:3])
            final_res = np.concatenate(final_res, axis=1)
            save_path = os.path.join(save_dir, str(beam_id) + '_subj_' + str(task_id) + '.csv')
            np.savetxt(save_path, final_res, delimiter=',')


def tmp_func():
    var_dict = torch.load("preliminary/MultiUN.var")
    var_res = var_dict['de-en']
    task_id = 3
    final_res = []
    for attack_id in [0, 1, 6]:
        res = torch.load('senstive/attack_type:' + str(attack_id) + '_model_type:'+ str(task_id) +'.sen')
        loops = res['beam'][4]
        print()
        var = [var_res[d[0]] for d in loops]
        var = np.sqrt(var)
        var = np.tile(np.array(var).reshape([-1, 1]), [1, 3])
        rate = (loops[:, 1:] - loops[:, 0:1]) / var
        res = [(rate > i).sum(0).reshape([1, -1]) for i in range(11)]
        res = np.concatenate(res)
        res = res / len(var) * 100
        final_res.append(res[:, 2:3])
    final_res = np.concatenate(final_res, axis=1)
    save_path = os.path.join(save_dir, 'subj_' + str(task_id) + '.csv')
    np.savetxt(save_path, final_res, delimiter=',')


if __name__ == '__main__':
    main()
    tmp_func()