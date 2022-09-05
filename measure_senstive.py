from utils import *
import torch
import argparse
import numpy as np
from tqdm import tqdm
device = torch.device('cuda')

MAX_PER = 3
MAX_LEN = 300

if not os.path.isdir('senstive'):
    os.mkdir('senstive')


def dump_model_info():
    for data_id in range(3):
        data_name = MODEL_NAME_LIST[data_id]
        model, tokenizer, _, _, _ = load_model(data_name)
        print(
            data_name,
            'vocab', tokenizer.vocab_size, '\n'
                                           'max_length', model.config.max_length, '\n',
            'num_beams', model.config.num_beams, '\n'
                                                 'num_beam_groups', model.config.num_beam_groups,
        )


def translate_config(model, tokenizer, adv_res, beam, group):
    res = []
    for i, adv_his in tqdm(enumerate(adv_res)):
        adv_his = adv_his[0:MAX_PER+1]
        out_len = []
        for (adv, _, _) in adv_his:
            input_token = tokenizer(adv, return_tensors="pt", padding=True).input_ids
            input_token = input_token.to(device)
            pred = model.generate(input_token, num_beams=beam, num_beam_groups=group)
            out_len.append(len(pred[0]))
        out_len = np.array(out_len).reshape([1, -1])
        res.append(out_len)
        if i == MAX_LEN:
            break
    return np.concatenate(res)


def main(model_id, attack_id):
    task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(model_id)
    data_name = MODEL_NAME_LIST[model_id]

    model, tokenizer, _, _, _ = load_model(data_name)
    model = model.to(device).eval()
    if model.config.max_length < 100:
        model.config.max_length = 200

    final_res = {'beam': []}
    for beam in [1, 2, 3, 4, 5]:
        adv_res = torch.load('adv/' + task_name + '_' + str(beam) + '.adv')
        group = model.config.num_beam_groups
        res = translate_config(model, tokenizer, adv_res, beam, group)
        final_res['beam'].append(res)
    torch.save(final_res, 'senstive/' + task_name + '.sen')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure Latency')
    parser.add_argument('--data', default=1, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack type')
    args = parser.parse_args()
    print('begin senstive measurement, data is %d, attack method is %d' %(args.data, args.attack))
    main(args.data, args.attack)
    exit(0)
