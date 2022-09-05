import torch
import numpy as np

from utils import *
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers.models.t5 import T5ForConditionalGeneration

#
# class Mobile(torch.nn.Module):
#     def __init__(self, net):
#         super(Mobile, self).__init__()
#         self.net = net
#
#     def forward(self, x):
#         return self.net.generate(x)
#
#
# attack_name = 'C'
# data_name = 'T5-small'
#
#
# benign_text = [
#     'death comes often to the soldiers and marines who are fighting in anbar province, which is roughly the size of louisiana and is the most intractable region in iraq.'
# ]
#
# adv_text = [
#     'death comes often to the soldiers and marines who are fighting in anbar province, which is roughly the (size of louisiana and is the most intractable region in iraq.'
# ]
#
# device = torch.device('cpu')
#
# model, tokenizer, _ = load_model(data_name)
# print()
#
#
# benign_token = tokenizer(benign_text, return_tensors="pt", padding=True).input_ids
# benign_token = benign_token.to(device)
# print(benign_token)
#
#
# adv_token = tokenizer(adv_text, return_tensors="pt", padding=True).input_ids
# adv_token = adv_token.to(device)
# print(adv_token)
#
#
# model.config.max_length = 200
# model.config.num_beams = 1

# benign_res = translate(model, benign_token)
# adv_res = translate(model, adv_token)
# print('translation finish')
#
# model = Mobile(model)
# res = model(benign_token)
# print('benign', res)
#
# script = torch.jit.trace(
#     model, example_inputs=benign_token
# )
# script.save("case_study/benign.pt")
# res = script.forward(benign_token)
# print('benign script', res)
#
#
# res = model(adv_token)
# print('adv', len(res))
#
# script = torch.jit.trace(
#     model, example_inputs=adv_token
# )
# script.save("case_study/adv.pt")
# res = script.forward(adv_token)
# print('adv script', len(res))

