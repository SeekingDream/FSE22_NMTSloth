import os
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


from src import *

BEAM_LIST = [4, 5, 1, 5]

MODEL_NAME_LIST = [
    'Helsinki-en-zh',
    'facebook-wmt19',
    'T5-small',
    'allenai-wmt16',

    'opus-mt-de-en'


]
MODEL_WEIGHT = 'model_weight'
ATTACKLIST = [
    CharacterAttack,
    WordAttack,

    Seq2SickAttack,
    NoisyAttack,

    SITAttack,
    TransRepairAttack,

    StructureAttack
]

if not os.path.isdir('res'):
    os.mkdir('res')
if not os.path.isdir(MODEL_WEIGHT):
    os.mkdir(MODEL_WEIGHT)


def load_model(model_name):
    if model_name == 'T5-small':
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'de'

    elif model_name == 'mbart-en-es':
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/mbart-large-finetuned-opus-en-es-translation")
        model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/mbart-large-finetuned-opus-en-es-translation")
        space_token = None
        src_lang, tgt_lang = 'en', 'es'

    elif model_name == 'Helsinki-en-zh':
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        space_token = '▁'
        src_lang, tgt_lang = 'en', 'zh'

    elif model_name == 'facebook-wmt19':
        tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de")
        space_token = '</w>'
        src_lang, tgt_lang = 'en', 'de'

    elif model_name == 'opus-mt-de-en':
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        space_token = '▁'
        src_lang, tgt_lang = 'de', 'en'

    elif model_name == 'allenai-wmt16':
        tokenizer = AutoTokenizer.from_pretrained("allenai/wmt16-en-de-dist-12-1")
        model = AutoModelForSeq2SeqLM.from_pretrained("allenai/wmt16-en-de-dist-12-1")
        space_token = '</w>'
        src_lang, tgt_lang = 'en', 'de'
    else:
        raise NotImplementedError
    return model, tokenizer, space_token, src_lang, tgt_lang


def load_dataset(model_name):
    if model_name == 'Helsinki-en-zh':
        with open('./data/Helsinki-en-zh.txt', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'facebook-wmt19':
        # with open('./data/rapid2019.txt', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'T5-small':
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'opus-mt-de-en':
        # with open('./data/wmt14.de', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data
    elif model_name == 'allenai-wmt16':
        # with open('./data/wmt14_valid.en', 'r') as f:
        #     data = f.readlines()
        #     return data
        with open('./data/translation2019zh/valid.en', 'r') as f:
            data = f.readlines()
            return data

    else:
        raise NotImplementedError


def load_model_dataset(model_name):
    model, tokenizer, space_token, src_lang, tgt_lang = load_model(model_name)
    dataset = load_dataset(model_name)
    return model, tokenizer, space_token, dataset, src_lang, tgt_lang


if __name__ == '__main__':
    m = load_model('allenai-wmt16')
    print()