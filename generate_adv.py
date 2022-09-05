import datetime
import os
import torch
import argparse


from utils import *


if not os.path.isdir('adv'):
    os.mkdir('adv')

MAX_TESTING_NUM = 500


def main(task_id, attack_id, beam):
    model_name = MODEL_NAME_LIST[task_id]
    # device = torch.device(7)
    device = torch.device('cuda')
    model, tokenizer, space_token, dataset, src_lang, tgt_lang = load_model_dataset(model_name)
    print('load model %s successful' % model_name)
    beam = model.config.num_beams if beam is None else beam
    config = {
        'num_beams': beam,
        'num_beam_groups': model.config.num_beam_groups,
        'max_per': 3,
        'max_len': 100,
        'src': src_lang,
        'tgt': tgt_lang
    }
    attack_class = ATTACKLIST[attack_id]
    attack = attack_class(model, tokenizer, space_token, device, config)
    task_name = 'attack_type:' + str(attack_id) + '_' + 'model_type:' + str(task_id)

    results = []
    t1 = datetime.datetime.now()
    for i, src_text in enumerate(dataset):
        if i == 0:
            continue
        if i >= MAX_TESTING_NUM:
            break
        # src_text = 'See gewohnt comprehensive for more information. </s>' # 'dataset[21]
        src_text = src_text.replace('\n', '')
        is_success, adv_his = attack.run_attack([src_text])
        if not is_success:
            print('error')
        for tmp in adv_his:
            assert type(tmp[0]) == str
            assert type(tmp[1]) == int
            assert type(tmp[2]) == float

        if len(adv_his) != config['max_per'] + 1:
            delta = config['max_per'] + 1 - len(adv_his)
            for _ in range(delta):
                adv_his.append(adv_his[-1])

        assert len(adv_his) == config['max_per'] + 1
        results.append(adv_his)
        torch.save(results, 'adv/' + task_name + '_' + str(beam) + '.adv')
    t2 = datetime.datetime.now()
    print(t2 - t1)
    torch.save(results, 'adv/' + task_name + '_' + str(beam) + '.adv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--data', default=3, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=6, type=int, help='attack type')
    parser.add_argument('--beam', default=None, type=int, help='beam size')
    args = parser.parse_args()
    main(args.data, args.attack, args.beam)
    exit(0)



