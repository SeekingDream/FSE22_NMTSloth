import datetime
import os
import torch
import argparse
import numpy as np

from utils import *


if not os.path.isdir('adv'):
    os.mkdir('adv')


def main():
    final_res = []
    for data_name in MODEL_NAME_LIST:
        for attack_type in [0, 1]:
            device = torch.device('cuda')
            model, tokenizer, space_token, dataset = load_model_dataset(data_name)
            print('load model %s successful' % data_name)
            config = {
                'num_beams': 1,
                'num_beam_groups': 1,
                'max_per': 5,
                'max_len': 100
            }
            if attack_type == 0:
                attack = CharacterAttack(model, tokenizer, space_token, device, config)
                attack_name = 'C'
            elif attack_type == 1:
                attack = WordAttack(model, tokenizer, space_token, device, config)
                attack_name = 'W'
            else:
                raise NotImplementedError
            results = []
            for i, src_text in enumerate(dataset):
                if i == 0:
                    continue
                if i >= 10:
                    break
                src_text = src_text.replace('\n', '')
                is_success, adv_his = attack.run_attack([src_text])
                if is_success:
                    overhead = np.array([r[-1] for r in adv_his]).reshape([1, -1])
                    results.append(overhead)
            try:
                results = np.concatenate(results)[:, 1:]
            except:
                print()
            mean, std = results.mean(0), results.std(0)
            final_res.append(np.array([mean, std]).reshape([1, -1]))
    final_res = np.concatenate(final_res)
    np.savetxt('sim/overhead.csv', final_res, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--data', default=2, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=1, type=int, help='attack type')
    args = parser.parse_args()
    main()
