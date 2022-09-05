import csv

import requests
from lxml import etree
import string
from tqdm import tqdm
import numpy as np
import json
import torch

PAGE_NUM = 49
MODEL_PER_PAGE = 30

final_dict = []
root_url = 'https://huggingface.co/models?pipeline_tag=translation&sort=downloads'


def parse_config(config):
    res = {}
    for key in config:
        if key == 'max_length':
            res['max_length'] = config['max_length']
        if key == 'max_time':
            res['max_time'] = config['max_time']
        if type(config[key]) == dict:
            new_res = parse_config(config[key])
            for k in new_res:
                res[k] = new_res[k]
    return res


def main():
    final_list = []
    cnt = 0
    f = open('config.csv', 'w')
    fwriter = csv.writer(f)
    for i in range(PAGE_NUM):
        if i == 0:
            base_url = root_url
        else:
            base_url = root_url + '&p=' + str(i)

        res = requests.get(base_url)
        root = etree.HTML(res.content)

        model_url_list = []
        for j in range(1, MODEL_PER_PAGE + 1):
            xpath = "/html/body/div/main/div/div/section[2]/div[3]/div/article[" + str(j) + "]/a/header/h4"
            model_name = root.xpath(xpath)
            try:
                model_url_list.append(model_name[0].text)
            except:
                pass
        base_url ='https://huggingface.co/'

        for model_name in tqdm(model_url_list):
            url = base_url + model_name
            res = requests.get(url)
            download_xpath = '/html/body/div/main/div/section[2]/div[1]/dl/dd'
            model_html = etree.HTML(res.content)
            download_num = model_html.xpath(download_xpath)
            download_num = int(download_num[0].text.replace(',', ''))
            url = base_url + model_name + '/resolve/main/config.json'
            res = requests.get(url)
            if res.status_code == 404:
                continue
            config = json.loads(res.text)
            res = parse_config(config)
            if 'max_length' not in res:
                res['max_length'] = 'None'
            if 'max_time' not in res:
                res['max_time'] = 'None'
            final_list.append([model_name, res['max_length'], res['max_time'], download_num])
            fwriter.writerow(final_list)
            cnt += 1
    f.close()
    torch.save(final_list, 'config.res')
    print(cnt, 'model parsed')


def post():
    results = torch.load('config.res')
    download_num = [d[3] for d in results]
    print(len(results), sum(download_num), max(download_num))

    results = [d for d in results if d[1] != 'None']
    print(len(results))
    f = open('config.csv', 'w')
    fwriter = csv.writer(f)
    fwriter.writerows(results)

    dict_res = {}
    for d in results:
        if d[1] not in dict_res:
            dict_res[d[1]] = 1
        else:
            dict_res[d[1]] += 1
    key_list = sorted(dict_res.keys())
    for k in key_list:
        fwriter.writerow([k, dict_res[k]])
        print(k, dict_res[k])

    f.close()

if __name__ == '__main__':
    post()