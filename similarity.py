import os

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import argparse

from utils import *

max_length = 50
eval_num = 500
device = torch.device('cuda')

if not os.path.isdir('sim'):
    os.mkdir('sim')


def compute_bleu(benign_token, adv_token, benign_pred, model):
    adv_pred = translate(
        model, adv_token,
        early_stopping=False, num_beams=1,
        num_beam_groups=1, use_cache=True,
        max_length=max_length
    )
    adv_pred = adv_pred['sequences']
    benign_token = benign_token.tolist()
    adv_token = adv_token.tolist()

    benign_pred = benign_pred.tolist()
    adv_pred = adv_pred.tolist()

    input_s = corpus_bleu([benign_token], adv_token)
    output_s = corpus_bleu([benign_pred], adv_pred)
    return input_s, output_s


def compute_similarity(data, model, tokenizer):
    input_res = [0, 0, 0, 0, 0]
    out_res = [0, 0, 0, 0, 0]
    benign = data[0][0]
    advs = [data[i][0] for i in range(1, len(data))]

    benign_token = tokenizer(benign, return_tensors="pt", padding=True).input_ids
    benign_token = benign_token.to(device)
    benign_pred = translate(
        model, benign_token,
        early_stopping=False, num_beams=1,
        num_beam_groups=1, use_cache=True,
        max_length=max_length
    )
    benign_pred = benign_pred['sequences']
    for i, adv in enumerate(advs):
        adv_token = tokenizer(adv, return_tensors="pt", padding=True).input_ids
        adv_token = adv_token.to(device)
        input_s, out_s = compute_bleu(benign_token, adv_token, benign_pred, model)
        input_res[i] = input_s
        out_res[i] = out_s
    return input_res, out_res


def main(attack_type):
    if attack_type == 0:
        attack_name = 'C'
    elif attack_type == 1:
        attack_name = 'W'
    else:
        raise NotImplementedError

    final_res = []
    for data_id in range(3):

        in_s = []
        out_s = []
        data_name = MODEL_NAME_LIST[data_id]
        model, tokenizer, _ = load_model(data_name)
        model = model.to(device)
        adv_res = torch.load('adv/' + attack_name + '_' + data_name + '.adv')
        adv_res = adv_res[:eval_num]
        for adv in tqdm(adv_res):
            input_res, out_res = compute_similarity(adv, model, tokenizer)
            in_s.append(np.array(input_res).reshape([1, -1]))
            out_s.append(np.array(out_res).reshape([1, -1]))
        in_s = np.concatenate(in_s).mean(0).reshape([-1, 1])
        out_s = np.concatenate(out_s).mean(0).reshape([-1, 1])
        res = np.concatenate([in_s, out_s], axis=1)
        final_res.append(res)
    final_res = np.concatenate(final_res)
    np.savetxt('sim/' + attack_name + '_similarity.csv', final_res, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--attack', default=1, type=int, help='attack type')
    args = parser.parse_args()
    main(args.attack)