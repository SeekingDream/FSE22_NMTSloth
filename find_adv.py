import torch
import numpy as np
import os


for file_name in os.listdir('adv'):
    if '.baseline' in file_name:
        continue
    adv_res = torch.load(os.path.join('adv', file_name))
    inc = np.array([r[1][1] /r[0][1] for r in adv_res])
    index = (-inc).argsort()
    c = 0
    print(file_name)
    print('-------------------------------------')
    for i in index:
        print(adv_res[i])
        c += 1
        if c == 10:
            break
