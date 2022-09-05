import os
import random
import time

from sklearn.svm import SVC
import torch
import numpy as np
from sklearn import metrics
import pyRAPL

from measure_latency import handle_cpu_energy
from utils import *

state_dir = 'hidden'
if not os.path.isdir(state_dir):
    os.mkdir(state_dir)


device = torch.device(7)


def collect_hidden(model_id, attack_id):
    def compute_length(s):
        return int(len(s) - sum(s.eq(tokenizer.pad_token_id)))

    data_name = MODEL_NAME_LIST[model_id]
    beam_size = BEAM_LIST[model_id]
    model, tokenizer, _, _, _ = load_model(data_name)
    task_name = 'attack_type:' + str(attack_id) + '_model_type:' + str(model_id)

    adv_res = torch.load('adv/' + task_name + '_' + str(beam_size) + '.adv')
    encoder = model.get_encoder().to(device).eval()

    benign_list = [d[0][0] for d in adv_res]
    adv_list = [d[-1][0] for d in adv_res if d[-1][0] != d[0][0]]
    benign_h_list, adv_h_list = [], []
    benign_len, adv_len = [], []
    batch_size = 10
    iter_num = len(benign_list) // batch_size
    if iter_num * batch_size != len(benign_list):
        iter_num = iter_num + 1
    for _ in range(iter_num):
        st, ed = batch_size * i, min(batch_size * (i + 1), len(benign_list))
        benign, adv = benign_list[st:ed], adv_list[st:ed]
        benign_tk = tokenizer(benign, return_tensors="pt", padding=True).input_ids.to(device)
        adv_tk = tokenizer(adv, return_tensors="pt", padding=True).input_ids.to(device)
        benign_h = encoder(benign_tk)
        adv_h = encoder(adv_tk)
        benign_h_list.append(benign_h.last_hidden_state.detach().cpu())
        adv_h_list.append(adv_h.last_hidden_state.detach().cpu())
        benign_len.extend([compute_length(s) for s in benign_tk])
        adv_len.extend([compute_length(s) for s in adv_tk])
    file_name = os.path.join(state_dir, str(model_id) + '_' + str(attack_id) + '.hidden')
    benign_h_list = torch.cat(benign_h_list)
    adv_h_list = torch.cat(adv_h_list)
    torch.save([(benign_h_list, adv_h_list), (benign_len, adv_len)], file_name)


def get_state(state, length, index):
    train_benign_h = state[index]
    train_benign_len = [length[i] for i in index]
    train_benign_h = [s[:train_benign_len[i]] for i, s in enumerate(train_benign_h)]
    return train_benign_h


def construct_xy(benign_hidden, adv_hidden):
    train_feature =\
        [d.mean(0).detach().cpu().numpy().reshape([1, -1]) for d in benign_hidden] +\
        [d.mean(0).detach().cpu().numpy().reshape([1, -1]) for d in adv_hidden]
    train_label = np.zeros([len(benign_hidden) + len(adv_hidden)])
    train_label[len(benign_hidden):] = 1
    train_feature = np.concatenate(train_feature)
    return train_feature, train_label


def train_detector(model_id, attack_id):
    file_name = os.path.join(state_dir, str(model_id) + '_' + str(attack_id) + '.hidden')
    [(benign_h_list, adv_h_list), (benign_len, adv_len)] = torch.load(file_name)
    all_num = len(benign_h_list)
    train_num = int(all_num * 0.8)
    train_index = random.sample(range(0, all_num), train_num)
    test_index = [i for i in range(all_num) if i not in train_index]

    train_benign_h = get_state(benign_h_list, benign_len, train_index)
    train_adv_h = get_state(adv_h_list, adv_len, train_index)

    test_benign_h = get_state(benign_h_list, benign_len, test_index)
    test_adv_h = get_state(adv_h_list, adv_len, test_index)

    train_feature, train_label = construct_xy(train_benign_h, train_adv_h)
    test_feature, test_label = construct_xy(test_benign_h, test_adv_h)

    m = SVC(probability=True)
    m.fit(train_feature, train_label)
    save_name = os.path.join(state_dir, str(model_id) + '_' + str(attack_id) + '.m')
    torch.save([m, (train_feature, train_label), (test_feature, test_label)], save_name)

    meter = pyRAPL.Measurement(save_name)
    meter.begin()
    t1 = time.time()
    test_pred = m.predict(test_feature)
    t2 = time.time()
    meter.end()
    latency = t2 - t1
    energy = handle_cpu_energy(meter.result.dram, meter.result.pkg)

    like_hod = m.predict_proba(test_feature)
    like_hod = like_hod[:, 1]
    acc = sum(test_pred == test_label) / len(test_label)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, like_hod, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return acc, auc, latency, energy


def train_test_mix(model_id):
    train_x, train_y, test_x, test_y = [], [], [], []
    for attack_id in [0, 1, 6]:
        save_name = os.path.join(state_dir, str(model_id) + '_' + str(attack_id) + '.m')
        [_, (train_feature, train_label), (test_feature, test_label)] = torch.load(save_name)
        train_x.append(train_feature)
        train_y.append(train_label)
        test_x.append(test_feature)
        test_y.append(test_label)
    train_x, train_y, test_x, test_y = \
        np.concatenate(train_x), np.concatenate(train_y), \
        np.concatenate(test_x), np.concatenate(test_y)

    m = SVC(probability=True)
    m.fit(train_x, train_y)

    meter = pyRAPL.Measurement(save_name)
    meter.begin()
    t1 = time.time()
    test_pred = m.predict(test_x)
    t2 = time.time()
    meter.end()
    latency = t2 - t1
    energy = handle_cpu_energy(meter.result.dram, meter.result.pkg)

    like_hod = m.predict_proba(test_x)
    like_hod = like_hod[:, 1]
    acc = sum(test_pred == test_y) / len(test_y)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, like_hod, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return acc, auc, latency, energy


if __name__ == '__main__':
    pyRAPL.setup()
    np.random.seed(101)
    final = []
    final_acc, final_auc, final_latency, final_energy = [], [], [], []
    for i in [0, 3, 2]:
        acc_score, auc_score, latency_score, energy_score = [], [], [], []
        for attack_id in [0, 1, 6]:
            # collect_hidden(i, attack_id)
            acc, auc, l, e = train_detector(i, attack_id)
            print(i, attack_id, 'successful', acc, auc)
            acc_score.append(acc)
            auc_score.append(auc)
            latency_score.append(l)
            energy_score.append(e)
        acc, auc, l, e = train_test_mix(i)
        acc_score.append(acc)
        auc_score.append(auc)
        latency_score.append(l)
        energy_score.append(e)

        acc_score = np.array(acc_score).reshape([1, -1])
        auc_score = np.array(auc_score).reshape([1, -1])
        latency_score = np.array(latency_score).reshape([1, -1])
        energy_score = np.array(energy_score).reshape([1, -1])
        final_acc.append(acc_score)
        final_auc.append(auc_score)
        final_latency.append(latency_score)
        final_energy.append(energy_score)

        tmp = np.concatenate([acc_score, auc_score, latency_score, energy_score])
        final.append(tmp)

    # final_acc = np.concatenate(final_acc)
    # final_auc = np.concatenate(final_auc)
    # final_latency = np.concatenate(final_latency)
    # final_energy = np.concatenate(final_energy)
    final = np.concatenate(final)
    np.savetxt('res/detector.csv', final, delimiter=',')


    # print('---------------------------------')
    # print('---------------------------------')
    # for i in range(4):
    #     for train_attack_id in [0, 1, 6]:
    #         save_name = os.path.join(state_dir, str(i) + '_' + str(train_attack_id) + '.m')
    #         [m, (_, _)] = torch.load(save_name)
    #         for test_attack_id in [0, 1, 6]:
    #             save_name = os.path.join(state_dir, str(i) + '_' + str(test_attack_id) + '.m')
    #             [_, (test_feature, test_label)] = torch.load(save_name)
    #             test_pred = m.predict(test_feature)
    #             acc = sum(test_pred == test_label) / len(test_label)
    #             print(i, train_attack_id, test_attack_id, acc)
    #
