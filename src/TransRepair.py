import time

import nltk
import numpy as np
import subprocess
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer


from .base_attack import SEAttack


def load_sim_dict():
    # load the similar word dictionary
    SIM_DICT_FILE = "/home/sxc180080/data/Project/TransAbuse/src/similarity_dict.txt"
    sim_dict = {}
    with open(SIM_DICT_FILE, 'r') as f:
        lines = f.readlines()
        for l in lines:
            sim_dict[l.split()[0]] = l.split()[1:]
    print("created dictionary")
    return sim_dict


class TransRepairAttack(SEAttack):

    def __init__(self, model, tokenizer, space_token, device, config):
        super(TransRepairAttack, self).__init__(model, tokenizer, space_token, device, config)

        self.tree_tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        self.similarity_threshold = 0.8
        self.sim_dict = load_sim_dict()

    def getLevenshtein(self, seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1])

    def normalizedED(self, seq1, seq2):
        dist = self.getLevenshtein(seq1, seq2)
        normalized_dist = 1 - (dist / max(len(seq1), len(seq2)))

        return normalized_dist

    def getSubSentSimilarity(self, subsentL1, subsentL2):
        similarity = -1

        for subsent1 in subsentL1:
            for subsent2 in subsentL2:
                currentSim = self.normalizedED(subsent1.split(' '), subsent2.split(' '))
                if currentSim > similarity:
                    similarity = currentSim

        return similarity

    def wordDiffSet(self, sentence1, sentence2):
        file1 = "temptest1.txt"
        file2 = "temptest2.txt"

        set1 = set()
        set2 = set()

        with open(file1, 'w') as f:
            f.write(sentence1)

        with open(file2, 'w') as f:
            f.write(sentence2)

        p = subprocess.run(["/home/sxc180080/bin/wdiff", file1, file2], stdout=subprocess.PIPE)
        wdstr = p.stdout.decode("utf-8")

        # print (wdstr)

        idxL1 = []
        idxL2 = []

        startIdx = -1
        endIdx = -1
        for idx, c in enumerate(wdstr):
            if c == '[':
                startIdx = idx
            elif c == ']':
                endIdx = idx
                idxL1.append((startIdx, endIdx))
            elif c == '{':
                startIdx = idx
            elif c == '}':
                endIdx = idx
                idxL2.append((startIdx, endIdx))

        for idxPair in idxL1:
            wordsS = wdstr[idxPair[0] + 2:idxPair[1] - 1]
            wordsL = wordsS.split(' ')
            set1 |= set(wordsL)

        for idxPair in idxL2:
            wordsS = wdstr[idxPair[0] + 2:idxPair[1] - 1]
            wordsL = wordsS.split(' ')
            set2 |= set(wordsL)

        return (set1, set2)

    def getSubSentenceList(self, sentence1, sentence2, set1, set2):
        # obtain the diff words
        # (set1, set2) = self.wordDiffSet(sentence1, sentence2)

        # generate sub sentences
        subsentL1 = []
        subsentL2 = []

        removeIdx1 = []
        removeIdx2 = []

        tokenizer = TreebankWordTokenizer()
        detokenizer = TreebankWordDetokenizer()

        sentence1L = tokenizer.tokenize(sentence1)
        sentence2L = tokenizer.tokenize(sentence2)

        for idx, word in enumerate(sentence1L):
            if word in set1:
                removeIdx1.append(idx)

        for idx, word in enumerate(sentence2L):
            if word in set2:
                removeIdx2.append(idx)

        for idx in removeIdx1:
            tokens = tokenizer.tokenize(sentence1)
            tokens.pop(idx)
            subsent = detokenizer.detokenize(tokens)
            subsentL1.append(subsent)

        for idx in removeIdx2:
            tokens = tokenizer.tokenize(sentence2)
            tokens.pop(idx)
            subsent = detokenizer.detokenize(tokens)
            subsentL2.append(subsent)

        return subsentL1, subsentL2

    def generate_sentences(self, sent):
        tokens = self.tree_tokenizer.tokenize(sent)
        pos_inf = nltk.tag.pos_tag(tokens)

        new_sentences, masked_indexes = [], []
        for idx, (word, tag) in enumerate(pos_inf):
            if word in self.sim_dict:
                masked_indexes.append((idx, tag))

        for (masked_index, tag) in masked_indexes:
            original_word = tokens[masked_index]
            # only replace noun, adjective, number
            if tag.startswith('NN') or tag.startswith('JJ') or tag == 'CD' or tag.startswith('PRP'):

                # generate similar sentences
                for similar_word in self.sim_dict[original_word]:
                    tokens[masked_index] = similar_word
                    new_pos_inf = nltk.tag.pos_tag(tokens)

                    # check that tag is still same type
                    if (new_pos_inf[masked_index][1].startswith(tag[:2])):
                        new_sentence = self.detokenizer.detokenize(tokens)
                        new_sentences.append(new_sentence)

                tokens[masked_index] = original_word

        return new_sentences

    def run_attack(self, origin_source_sent):
        assert len(origin_source_sent) == 1
        t1 = time.time()
        origin_target_sent, original_target_len = self.get_trans_strings(origin_source_sent)

        origin_target_sent = origin_target_sent[0]
        original_target_len = original_target_len[0]

        adv_his = [(origin_source_sent[0], int(original_target_len), 0.0)]
        count = 0

        new_sentsL = self.generate_sentences(origin_source_sent[0])
        if len(new_sentsL) == 0:
            return False, adv_his

        new_target_sent_sL, new_target_lensL = self.get_trans_strings(new_sentsL)
        for new_source, new_target, new_len in zip(new_sentsL, new_target_sent_sL, new_target_lensL):
            # obtain the segmented one for Chinese
            sentence1 = self.split_token(origin_target_sent)
            sentence2 = self.split_token(new_target)
            # obtain different words by wdiff
            set1, set2 = self.wordDiffSet(sentence1, sentence2)
            # get sub sentences
            subsentL1, subsentL2 = self.getSubSentenceList(sentence1, sentence2, set1, set2)

            similarity = self.getSubSentSimilarity(subsentL1, subsentL2)

            if similarity != -1 and similarity < self.similarity_threshold:
                # suspicous_issueL.append(
                #     (str(count), similarity, origin_source_sent, origin_target_sent, new_source, new_target))
                count += 1
                t2 = time.time()
                adv_his.append((new_source, int(new_len), t2 - t1))
            if count == self.max_per:
                break
        return True, adv_his
