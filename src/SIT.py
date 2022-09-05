from nltk.parse import CoreNLPDependencyParser
import jieba
import nltk
import Levenshtein
from nltk.data import find
import numpy as np
from google.cloud import translate
import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import time
import pickle
import os, requests, uuid, json

from .base_attack import SEAttack
from .TranslateAPI import translate

nltk.download('averaged_perceptron_tagger')


class SITAttack(SEAttack):
	def __init__(self, model, tokenizer, space_token, device, config):
		super(SITAttack, self).__init__(model, tokenizer, space_token, device, config)
		# initialize the dependency parser

		target_port = self.port_dict[self.target_language]
		self.chi_parser = CoreNLPDependencyParser('http://localhost:' + str(target_port))

		# use nltk treebank tokenizer and detokenizer
		self.tree_tokenizer = TreebankWordTokenizer()
		self.detokenizer = TreebankWordDetokenizer()

		# BERT initialization
		self.berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
		self.bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
		self.bertmodel = self.bertmodel.eval().to(self.model.device)

		# parameters
		self.num_of_perturb = 10  # number of generated similar words for a given position
		self.distance_threshold = 0.0  # the distance threshold in "translation error detection via structure comparison"
		self.issue_threshold = self.max_per  # how many output issues
		self.sentenceBount = 10000  # an upperbound to avoid translating too many sentences
		# self.output_file = 'results_' + dataset + '_' + software + '.txt'
		# self.write_output = open(self.output_file, 'w')

		self.sent_count = 0
		self.issue_count = 0

	# Generate a list of similar sentences by Bert
	def perturb(self, sent):
		tokens = self.tree_tokenizer.tokenize(sent[0])
		pos_inf = nltk.tag.pos_tag(tokens)

		# the elements in the lists are tuples <index of token, pos tag of token>
		bert_masked_indexL = list()

		# collect the token index for substitution
		for idx, (word, tag) in enumerate(pos_inf):
			# substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
			if tag.startswith('NN') or tag.startswith('JJ'):
				tagFlag = tag[:2]
				# we do not perturb the first and the last token because BERT's performance drops on for those positions
				if idx != 0 and idx != len(tokens) - 1:
					bert_masked_indexL.append((idx, tagFlag))

		bert_new_sentences = list()

		# generate similar setences using Bert
		if bert_masked_indexL:
			bert_new_sentences = self.perturbBert(sent, bert_masked_indexL)

		return bert_new_sentences

	def perturbBert(self, sent, masked_indexL):
		# self.bertmodel, self.num_of_perturb,
		new_sentences = list()
		tokens = self.tree_tokenizer.tokenize(sent[0])

		invalidChars = set(string.punctuation)

		# for each idx, use Bert to generate k (i.e., num) candidate tokens
		for (masked_index, tagFlag) in masked_indexL:
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
				break

			# get the similar words
			topk_Idx = torch.topk(prediction[0][0, masked_index], self.num_of_perturb)[1].tolist()
			topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)

			# remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
			# this step could be further optimized by filtering more tokens (e.g., non-english tokens)
			topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

			# generate similar sentences
			for t in topk_tokens:
				if any(char in invalidChars for char in t):
					continue
				tokens[masked_index] = t
				new_pos_inf = nltk.tag.pos_tag(tokens)

				# only use the similar sentences whose similar token's tag is still NN or JJ
				if (new_pos_inf[masked_index][1].startswith(tagFlag)):
					new_sentence = self.detokenizer.detokenize(tokens)
					new_sentences.append(new_sentence)

			tokens[masked_index] = original_word

		return new_sentences

	# calculate the distance between
	def depDistance(self, graph1, graph2):
		# count occurences of each type of relationship
		counts1 = dict()
		for i in graph1:
			counts1[i[1]] = counts1.get(i[1], 0) + 1

		counts2 = dict()
		for i in graph2:
			counts2[i[1]] = counts2.get(i[1], 0) + 1

		all_deps = set(list(counts1.keys()) + list(counts2.keys()))
		diffs = 0
		for dep in all_deps:
			diffs += abs(counts1.get(dep, 0) - counts2.get(dep, 0))
		return diffs

	def run_attack(self, origin_source_sent):
		t1 = time.time()
		assert len(origin_source_sent) == 1
		origin_target_sent, origin_pred_len = self.get_trans_strings(origin_source_sent)

		origin_target_sent = origin_target_sent[0]
		origin_pred_len = origin_pred_len[0]

		target_sent_seg = self.split_token(origin_target_sent)
		origin_target_tree_Ls = self.chi_parser.raw_parse_sents(
			[target_sent_seg], properties={'ssplit.eolonly': 'true'}
		)   # todo
		origin_target_tree = [target_tree for (target_tree,) in origin_target_tree_Ls]
		origin_target_tree = origin_target_tree[0]
		suspicious_issues = list()
		new_source_sentsL = self.perturb(origin_source_sent)

		adv_his = [(origin_source_sent[0], int(origin_pred_len), 0.0)]

		if len(new_source_sentsL) == 0:
			return False, adv_his

		new_target_sents_segL = list()
		new_target_sentsL, new_target_lensL = self.get_trans_strings(new_source_sentsL)
		for new_target_sent in new_target_sentsL:

			new_target_sent_seg = self.split_token(new_target_sent)
			new_target_sents_segL.append(new_target_sent_seg)
		tmp_res = self.chi_parser.raw_parse_sents(
				new_target_sents_segL,
				properties={'ssplit.eolonly': 'true'}
			)
		new_target_treesL = [target_tree for (target_tree,) in tmp_res]
		assert (len(new_target_treesL) == len(new_source_sentsL))
		# print('new target sentences parsed')

		for (new_source_sent, new_target_sent, new_target_len, new_target_tree) in \
				zip(new_source_sentsL, new_target_sentsL, new_target_lensL, new_target_treesL):
			distance = self.depDistance(origin_target_tree.triples(), new_target_tree.triples())

			if distance > self.distance_threshold:
				suspicious_issues.append((new_source_sent, new_target_sent, new_target_len, distance))
		# print('distance calculated')

		# clustering by distance for later sorting
		suspicious_issues_cluster = dict()
		for (new_source_sent, new_target_sent, new_target_len, distance) in suspicious_issues:
			if distance not in suspicious_issues_cluster:
				new_cluster = [(new_source_sent, new_target_sent, new_target_len)]
				suspicious_issues_cluster[distance] = new_cluster
			else:
				suspicious_issues_cluster[distance].append((new_source_sent, new_target_sent, new_target_len))
		# print('clustered')

		# if no suspicious issues
		if len(suspicious_issues_cluster) == 0:
			return False, adv_his

		# sort by distance, from large to small
		sorted_keys = sorted(suspicious_issues_cluster.keys())
		sorted_keys.reverse()

		remaining_issue = self.issue_threshold
		# Output the top k sentences

		suspicious_issues = [suspicious_issues_cluster[k] for k in sorted_keys]
		res = []
		for issue in suspicious_issues:
			res.extend(issue)

		for r in res:
			t2 = time.time()
			adv_his.append((r[0], int(r[2]), t2 - t1))
			remaining_issue -= 1
			if remaining_issue == 0:
				break

		# for distance in sorted_keys:
		# 	if remaining_issue == 0:
		# 		break
		# 	candidateL = suspicious_issues_cluster[distance]
		# 	sortedL = sorted(candidateL, key=lambda x: len(x[1]))
		# 	issue_threshold_current = remaining_issue
		# 	for i in range(issue_threshold_current):
		# 		remaining_issue -= 1
		# 		t2 = time.time()
		# 		adv_his.append((sortedL[i][0], sortedL[i][2], t2 - t1))
		#
		# 		# return sortedL[i][0], sortedL[i][1]
		return True, adv_his


			# if len(candidateL) <= remaining_issue:
			# 	remaining_issue -= len(candidateL)
			# 	for candidate in candidateL:
			# 		return candidate[0], candidate[1]  # todo
			#
			# else:


