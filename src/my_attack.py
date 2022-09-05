import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import nltk
import string
from copy import deepcopy
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from .base_attack import MyAttack


class CharacterAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(CharacterAttack, self).__init__(model, tokenizer, space_token, device, config)
        # self.eos_token_id = self.model.config.eos_token_id
        # self.pad_token_id = self.model.config.pad_token_id
        #
        # self.num_beams = config['num_beams']
        # self.num_beam_groups = config['num_beam_groups']
        # self.max_per = config['max_per']
        # self.embedding = self.model.get_input_embeddings().weight
        # self.softmax = nn.Softmax(dim=1)
        # self.bce_loss = nn.BCELoss()
        # self.specical_token = self.tokenizer.all_special_tokens
        # self.space_token = space_token
        # self.max_len = config['max_len']
        # self.insert_character = string.punctuation
        # self.insert_character += string.digits
        # self.insert_character += string.ascii_letters

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def mutation(self, current_adv_text, grad, modify_pos):
        current_tensor = self.tokenizer([current_adv_text], return_tensors="pt", padding=True).input_ids[0]
        new_strings = self.character_replace_mutation(current_adv_text, current_tensor, grad)
        return new_strings

    @staticmethod
    def transfer(c: str):
        if c in string.ascii_lowercase:
            return c.upper()
        elif c in string.ascii_uppercase:
            return c.lower()
        return c

    def character_replace_mutation(self, current_text, current_tensor, grad):
        important_tensor = (-grad.sum(1)).argsort()
        # current_string = [self.tokenizer.decoder[int(t)] for t in current_tensor]
        new_strings = [current_text]
        for t in important_tensor:
            if int(t) not in current_tensor:
                continue
            ori_decode_token = self.tokenizer.decode([int(t)])
            if self.space_token in ori_decode_token:
                ori_token = ori_decode_token.replace(self.space_token, '')
            else:
                ori_token = ori_decode_token
            if len(ori_token) == 1 or ori_token in self.specical_token:  #todo
                continue
            candidate = [ori_token[:i] + insert + ori_token[i:] for i in range(len(ori_token)) for insert in self.insert_character]
            candidate += [ori_token[:i - 1] + self.transfer(ori_token[i - 1]) + ori_token[i:] for i in range(1, len(ori_token))]

            new_strings += [current_text.replace(ori_token, c, 1) for c in candidate]

            # ori_tensor_pos = current_tensor.eq(int(t)).nonzero()
            #
            # for p in ori_tensor_pos:
            #     new_strings += [current_string[:p] + c + current_string[p + 1:] for c in candidate]
            if len(new_strings) != 0:
                return new_strings
        return new_strings


class WordAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(WordAttack, self).__init__(model, tokenizer, space_token, device, config)

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)
        return new_strings

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def token_replace_mutation(self, current_adv_text, grad, modify_pos):
        new_strings = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        base_tensor = current_tensor.clone()
        for pos in modify_pos:
            t = current_tensor[0][pos]
            grad_t = grad[t]
            score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
            index = score.argsort()
            for tgt_t in index:
                if tgt_t not in self.specical_token:
                    base_tensor[pos] = tgt_t
                    break
        current_text = self.tokenizer.decode(current_tensor)
        for pos, t in enumerate(current_tensor):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        new_base_tensor = base_tensor.clone()
                        new_base_tensor[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_tensor)
                        # if new_tag[pos][:2] == ori_tag[pos][:2]:
                        new_strings.append(candidate_s)
                        cnt += 1
                        if cnt >= 50:
                            break
        return new_strings


class StructureAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(StructureAttack, self).__init__(model, tokenizer, space_token, device, config)
        # self.tree_tokenizer = TreebankWordTokenizer()
        # self.detokenizer = TreebankWordDetokenizer()
        # BERT initialization
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.bertmodel = bertmodel.eval().to(self.model.device)
        self.num_of_perturb = 50

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def perturbBert(self, tokens, ori_tensors, masked_indexL, masked_index):
        new_sentences = list()
        # invalidChars = set(string.punctuation)

        # for each idx, use Bert to generate k (i.e., num) candidate tokens
        original_word = tokens[masked_index]

        low_tokens = [x.lower() for x in tokens]
        low_tokens[masked_index] = '[MASK]'
        # try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(self.model.device)
            prediction = self.bertmodel(tokens_tensor)

            # skip the sentences that contain unknown words
            # another option is to mark the unknow words as [MASK]; we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            return new_sentences

        # get the similar words
        topk_Idx = torch.topk(prediction[0][0, masked_index], self.num_of_perturb)[1].tolist()
        topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)

        # remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # this step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

        # generate similar sentences
        for t in topk_tokens:
            # if any(char in invalidChars for char in t):
            #     continue
            tokens[masked_index] = t
            new_pos_inf = nltk.tag.pos_tag(tokens)

            # only use the similar sentences whose similar token's tag is still the same
            if new_pos_inf[masked_index][1][:2] == masked_indexL[masked_index][1][:2]:
                new_t = self.tokenizer.encode(tokens[masked_index])[0]
                new_tensor = ori_tensors.clone()
                new_tensor[masked_index] = new_t
                new_sentence = self.tokenizer.decode(new_tensor)
                new_sentences.append(new_sentence)
        tokens[masked_index] = original_word
        return new_sentences

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.structure_mutation(current_adv_text, grad)
        return new_strings

    def get_token_type(self, input_tensor):
        # tokens = self.tree_tokenizer.tokenize(sent)
        tokens = self.tokenizer.convert_ids_to_tokens(input_tensor)
        tokens = [tk.replace(self.space_token, '') for tk in tokens]
        pos_inf = nltk.tag.pos_tag(tokens)
        bert_masked_indexL = list()
        # collect the token index for substitution
        for idx, (word, tag) in enumerate(pos_inf):
            # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
            # if tag.startswith('NN') or tag.startswith('JJ'):
            #     tagFlag = tag[:2]
                # we do not perturb the first and the last token because BERT's performance drops on for those positions
            # if idx != 0 and idx != len(tokens) - 1:
            bert_masked_indexL.append((idx, tag))

        return tokens, bert_masked_indexL

    def structure_mutation(self, current_adv_text, grad):
        new_strings = []
        important_tensor = (-grad.sum(1)).argsort()

        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]

        ori_tokens, ori_tag = self.get_token_type(current_tensor)
        assert len(ori_tokens) == len(current_tensor)
        assert len(ori_tokens) == len(ori_tag)
        current_tensor_list = current_tensor.tolist()
        for t in important_tensor:
            if int(t) not in current_tensor_list:
                continue
            pos_list = torch.where(current_tensor.eq(int(t)))[0].tolist()
            for pos in pos_list:
                new_string = self.perturbBert(ori_tokens, current_tensor, ori_tag, pos)
                new_strings.extend(new_string)
            if len(new_strings) > 2000:
                break
        return new_strings
