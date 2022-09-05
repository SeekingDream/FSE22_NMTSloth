import os
import torch
import time
from utils import MODEL_NAME_LIST, load_model, translate
from tqdm import tqdm
import numpy as np

if not os.path.isdir('transfer_res'):
    os.mkdir('transfer_res')

@torch.no_grad()
def measure_transferbility(model, tokenizer, adv_his, max_length, device):
    metric = {
        'flops': [],
    }
    model = model.to(device)
    for i, (adv, _) in enumerate(adv_his):
        input_token = tokenizer(adv, return_tensors="pt", padding=True).input_ids
        input_token = input_token.to(device)
        pred = translate(
            model, input_token,
            early_stopping=False, num_beams=1,
            num_beam_groups=1, use_cache=True,
            max_length=max_length
        )
        metric['flops'].append(len(pred['sequences'][0]))
    return metric['flops']


def main():
    device = torch.device(7)
    max_length = 200
    for attack_name in ['C', 'W']:
        for model_name in MODEL_NAME_LIST:
            model, tokenizer, _ = load_model(model_name)
            for data_name in MODEL_NAME_LIST:
                if data_name == model_name:
                    continue

                adv_res = torch.load('adv/' + attack_name + '_' + data_name + '.adv')
                save_path = os.path.join('transfer_res', attack_name + '_' + model_name + '_' + data_name + '.res')
                final_res_data = []
                # ori_flops = np.array([d[0][1] for d in adv_res])
                for adv in tqdm(adv_res):
                    ori_flops = adv[0][1]
                    metric = measure_transferbility(model, tokenizer, adv, max_length, device)
                    final_res_data.append(((np.array(metric) / ori_flops) - 1).reshape([1, -1]))
                    torch.save(final_res_data, save_path)
                    if len(final_res_data) > 500:
                        break
                print(attack_name, model_name, data_name, 'successful')
                print('mean', np.concatenate(final_res_data).mean(0))
                print('max', np.concatenate(final_res_data).max(0))


def post():
    for attack_name in ['C', 'W']:
        final_res = []
        for i, model_name in enumerate(MODEL_NAME_LIST):
            for j, data_name in enumerate(MODEL_NAME_LIST):
                if data_name == model_name:
                    continue
                save_path = os.path.join('transfer_res', attack_name + '_' + model_name + '_' + data_name + '.res')
                res = torch.load(save_path)
                res = np.concatenate(res).max(0)
                res = res * 100
                res = np.array([(i + 1) * 1000 + (j + 1)] + list(res))
                final_res.append(res.reshape([1, -1]))
        final_res = np.concatenate(final_res)
        np.savetxt('res/trans_' + attack_name + '.csv', final_res, delimiter=',')


if __name__ == '__main__':
    # main()
    post()
