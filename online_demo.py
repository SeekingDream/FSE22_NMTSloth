import datetime
import os
import torch
import argparse

from utils import *

if not os.path.isdir('adv'):
    os.mkdir('adv')

MAX_TESTING_NUM = 1000


def main():
    task_id = 3
    attack_id = 0
    model_name = MODEL_NAME_LIST[task_id]
    device = torch.device(7)
    model, tokenizer, space_token, dataset, src_lang, tgt_lang = load_model_dataset(model_name)
    print('load model %s successful' % model_name)
    config = {
        'num_beams': model.config.num_beams,
        'num_beam_groups': model.config.num_beam_groups,
        'max_per': 5,
        'max_len': 100,
        'src': src_lang,
        'tgt': tgt_lang
    }
    max_per = config['max_per']
    attack_class = ATTACKLIST[attack_id]
    attack = attack_class(model, tokenizer, space_token, device, config)
    task_name = 'attack_type:' + str(attack_id) + '_' + 'model_type:' + str(task_id)

    best_inc = [1 for _ in range(max_per)]
    best_seed = [None for _ in range(max_per)]
    results = []
    t1 = datetime.datetime.now()
    for i, src_text in enumerate(dataset):
        if i >= MAX_TESTING_NUM:
            break
        src_text = src_text.replace('\n', '')
        is_success, adv_his = attack.run_attack([src_text])
        # adv_his = List[[sentence], len, overheads]
        results.append(adv_his)
        inc = [adv_his[i][1] / adv_his[0][1] for i in range(1, len(adv_his))]
        for jjj in range(max_per):
            if inc[jjj] > best_inc[jjj]:
                best_inc[jjj] = inc[jjj]
                best_seed[jjj] = [
                    [adv_his[0][1], adv_his[jjj][1]],
                    [adv_his[0][0], adv_his[jjj][0]]
                ]
        print()
        torch.save(results, 'adv/demo.adv')
    for i in range(len(best_inc)):
        print(best_inc[i], best_seed[i])
    t2 = datetime.datetime.now()
    print(t2 - t1)
    torch.save(results, 'adv/demo.adv')


if __name__ == '__main__':
    main()
