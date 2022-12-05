import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from typing import List
import random
import argparse
import copy
import h5py
import time
import numpy as np
import math
import torch
import time
# print(torch.cuda.is_available())
import sys
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

sys.path.append('/Users/teddy/PycharmProjects/BERT-WSD')
sys.path.append('/disk/scratch_big/tli/BERT-WSD')
sys.path.append('../utils')
import re
from transformers import BertTokenizer
from script.utils.dataset import GlossSelectionRecord, _create_features_from_records, collate_batch
from script.utils.model import BertWSD, forward_gloss_selection
from script.utils.wordnet import get_glosses
from torch.nn.functional import softmax

from queue import PriorityQueue
from qaeval_utils import DateManager, parse_rel, readWordNet, upred2bow, rel2concise_str, assemble_rel, load_triple_set, \
	readWord2Vec, build_upred_vectors_h5py, calc_simscore, truncate_merge_by_ratio, filter_sound_triples, \
	check_vague, duration_format_print, reconstruct_sent_from_rel, lemmatize_str, all_toks_are_preps, fits_thresholds

# print(torch.cuda.is_available())

accepted_neg_indicator_reltypes = ['hypo']


def get_predictions(model, tokenizer, sentence, device_name, batch_size):
	re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence)
	if re_result is None:
		print("\nIncorrect input format. Please try again.")
		return

	ambiguous_word = re_result.group(1).strip()
	sense_keys = []
	definitions = []
	for sense_key, definition in get_glosses(ambiguous_word, None).items():
		sense_keys.append(sense_key)
		definitions.append(definition)

	record = GlossSelectionRecord("test", sentence, sense_keys, definitions, [-1])
	features = _create_features_from_records([record], 128, tokenizer,
											 cls_token=tokenizer.cls_token,
											 sep_token=tokenizer.sep_token,
											 cls_token_segment_id=1,
											 pad_token_segment_id=0,
											 disable_progress_bar=True)[0]
	collated_features = []
	subfeatures = []
	for f in features:
		subfeatures.append(f)

		if len(subfeatures) == batch_size:
			collated_features.append(subfeatures)
			subfeatures = []
		else:
			assert len(subfeatures) < batch_size
	if len(subfeatures) > 0:
		collated_features.append(
			subfeatures)  # append the last chunk to collated_features, however many number of entries (but certainly <= batch_size)

	# 0: input_ids; 1: input_mask; 2: segment_ids; 3: label_id
	collated_features = collate_batch(collated_features) if len(features) > 0 else []

	with torch.no_grad():
		logits = torch.zeros(len(definitions), dtype=torch.double).to(device_name)
		for batch_i, bert_input in enumerate(collated_features):
			bert_output = model.bert(input_ids=bert_input[0].to(device_name),
									attention_mask=bert_input[1].to(device_name),
									token_type_ids=bert_input[2].to(device_name))
			cur_batch_logits = model.ranking_linear(bert_output[1])
			for in_batch_i, logi in enumerate(cur_batch_logits):
				global_i = batch_i * batch_size + in_batch_i
				logits[global_i] = logi
		scores = softmax(logits, dim=0)

	return sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True)


def compute_posi_synsets(potential_positives_path: str, wsd_model_dir: str, date_slices: List[str], batch_size: int,
						 lang: str, date_slice_idx: int = None, no_cuda: bool = False):
	# single words often have multiple senses, where it would be good to do WSD; for multi-word collocations, it is less
	# common that multiple senses are associated with it, therefore we don't do WSD for that.

	model = BertWSD.from_pretrained(wsd_model_dir)
	tokenizer = BertTokenizer.from_pretrained(wsd_model_dir)
	print(f"torch cuda is_available: {torch.cuda.is_available()}")
	device_name = "cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu"
	print(f"device name: {device_name}")
	model.to(device_name)
	model.eval()

	# these stop words have no WordNet entries, so don't bother running the algorithm for them.
	stop_words = ['in', 'on', 'at', 'with', 'about', 'from', 'to', 'for', 'of', "'s", "'", "by"]

	missing_predtoks = dict()

	for cur_slice_idx, dslc in enumerate(date_slices):
		if date_slice_idx is not None and cur_slice_idx != date_slice_idx:
			continue
		print(f"current date slice: {dslc}")
		# if dslc != '2008_01-22_01-24':
		# 	continue
		cur_path = potential_positives_path % dslc

		cur_entries = []

		if not os.path.exists(cur_path):
			print(f"path does not exist: {cur_path}; continueing...")
			continue

		with open(cur_path, 'r', encoding='utf8') as rfp:
			pp_lidx = 0
			for pp_lidx, potpos_line in enumerate(rfp):
				if pp_lidx % 1000 == 0:
					print(pp_lidx)
					print(f"lidx: {pp_lidx}; number of unique missing tokens: {len(missing_predtoks)};")

				if len(potpos_line) < 2:
					continue
				potpos_item = json.loads(potpos_line)
				upred, subj, obj, tsubj, tobj = parse_rel(potpos_item)
				_, pred_str = rel2concise_str(upred, subj, obj, tsubj, tobj, lang)
				pred_str = pred_str.split(' ')
				potpos_item['wn_synsets'] = {}
				for tid, tok in enumerate(pred_str):
					if tok in stop_words:
						potpos_item['wn_synsets'][tok] = []
						continue
					pseudo_sent = ' '.join(
						[subj] + pred_str[:tid] + ['[TGT]', pred_str[tid], '[TGT]'] + pred_str[tid + 1:] + [obj])
					predictions = get_predictions(model, tokenizer, pseudo_sent, device_name, batch_size)
					curtok_synsets = []
					# Only take the top 3 synsets with predicted scores larger than 0.01.
					for (key, _, score) in predictions:
						if len(curtok_synsets) >= 3:
							break
						elif score < 0.03:
							continue
						else:
							curtok_synsets.append(key)
					if len(curtok_synsets) == 0:
						if tok not in missing_predtoks:
							missing_predtoks[tok] = 0
						missing_predtoks[tok] += 1
					potpos_item['wn_synsets'][tok] = curtok_synsets
				cur_entries.append(potpos_item)
			print(f"lidx: {pp_lidx}; number of unique missing tokens: {len(missing_predtoks)};")

		with open(cur_path, 'w', encoding='utf8') as ofp:
			for potpos_item in cur_entries:
				out_line = json.dumps(potpos_item, ensure_ascii=False)
				ofp.write(out_line + '\n')

	missing_predtoks = [k for k, v in sorted(missing_predtoks.items(), key=lambda x: x[1], reverse=True)]
	print(f"Missing pred_toks: ")
	print(missing_predtoks)


def find_negatives_wordnet(positives, wordnet_dict, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr, lemm_noprep_2_predstr,
						   partition_triples_path: str, vnonly: bool, firstonly: bool, global_presence: str, global_presence_thres: int,
						   global_presence_cap: int, global_triple_absence_flag: bool,
						   max_num_posi_collected_per_partition, verbose, lang, allow_prep_mismatch_replacements,
						   allow_backoff_wsd):

	assert global_presence in ['triple', 'pred']

	lemmatizer = WordNetLemmatizer()

	if lang == 'zh':
		assert wordnet_dict is not None
		lexicalEntries = wordnet_dict['lexicalEntries']
		synsets = wordnet_dict['synsets']
		sstids2lemmas = wordnet_dict['sstids2lemmas']
		zh_en_alignments = wordnet_dict['zh_en_alignments']
	elif lang == 'en':
		assert wordnet_dict is None
		lexicalEntries = None
		synsets = None
		sstids2lemmas = None
		zh_en_alignments = None
	else:
		raise AssertionError

	potential_negatives = []
	num_negs_found = []

	all_partition_keys = set()
	for posi_entry in positives:
		all_partition_keys.add(posi_entry['partition_key'])
	all_partition_keys = list(all_partition_keys)
	all_partition_keys.sort()

	processed_posi_idxes = set()

	for pk in all_partition_keys:
		if global_triple_absence_flag:
			assert global_tplstrs is not None
			cur_tplstrs2exclude = global_tplstrs
		else:
			cur_partition_triples, cur_partition_preds = load_triple_set(partition_triples_path % pk, verbose=False,
																		 lang=lang)
			cur_tplstrs2exclude = set(cur_partition_triples.keys())
			del cur_partition_triples
			del cur_partition_preds
		# print(cur_tplstrs2exclude)
		assert isinstance(cur_tplstrs2exclude, set)
		print(f"fetching negatives for positives in partition: {pk}")

		# below is the set of sentence indices where some negatives can be found from some of the positives in this sentence.
		# the size of this set is the same as the number of positives attended in this partition, since we are going to
		# dictate that only one positive per sentence can be selected into the pool.
		negfound_sidxes = set()
		all_posi_idxes = list(range(len(positives)))
		random.shuffle(all_posi_idxes)
		cur_partition_num_posis_processed = 0

		for posi_idx in all_posi_idxes:
			posi_entry = positives[posi_idx]

			if cur_partition_num_posis_processed % 10000 == 0:
				print(f"cur_partition_num_posis_processed: {cur_partition_num_posis_processed}")

			# in each round, only process the entries in a single partition, this is to control memory usage.
			if posi_entry['partition_key'] != pk:
				continue
			cur_partition_num_posis_processed += 1

			assert posi_idx not in processed_posi_idxes
			processed_posi_idxes.add(posi_idx)

			# if there has been other positives selected from this sentence, then no other positives are considered.
			if posi_entry['in_partition_sidx'] in negfound_sidxes:
				continue

			if len(negfound_sidxes) > max_num_posi_collected_per_partition:
				print(f"Maximum number of positives to collect reached for current partition: ``{pk}''")
				break

			upred, subj, obj, tsubj, tobj = parse_rel(posi_entry)
			posi_tplstr, posi_predstr = rel2concise_str(upred, subj, obj, tsubj, tobj, lang=lang)
			assert posi_tplstr in cur_tplstrs2exclude
			upred_list = upred2bow(upred, lang=lang)
			if lang == 'en':
				tok_synsets = posi_entry['wn_synsets']
			elif lang == 'zh':
				tok_synsets = None
			else:
				raise AssertionError

			if check_vague(upred_list) is True:
				continue

			type_pair = f"{tsubj}#{tobj}"
			neg_predstrs_after_filter = set()

			negfound_predstr_spans = []

			for start_id in range(len(upred_list)):
				for end_id in range(len(upred_list), start_id, -1):
					# Enumerate the spans:
					# 		By enumerating continuous spans, naturally parts of the predicates split by 'X'
					# 		will not be taken as consecutive strings.

					#		Those sub-spans of those already associated-with-negatives spans should not be attended anymore.
					#		This is enforced by maintaining a list of negfound_spans, then continueing when a span is a subspan of any of those.

					subspan_of_good_spans_flag = False  # if this flag is true, then there is no point in attending to the current span!
					for negfound_span in negfound_predstr_spans:
						if start_id >= negfound_span['start_id'] and end_id <= negfound_span['end_id']:
							subspan_of_good_spans_flag = True
							break
					if subspan_of_good_spans_flag:
						continue

					if lang == 'zh':
						cur_span = ''.join(upred_list[start_id:end_id])
						prefix = ''.join(upred_list[:start_id])
						postfix = ''.join(upred_list[end_id:])
						if cur_span not in lexicalEntries:
							continue
						potneg_replacements = {}
						cur_lex_ents = lexicalEntries[cur_span]
						for cur_lex_ent in cur_lex_ents:
							# Enumerate all entries with this surface form

							# vnonly: only take those WordNet entries with 'v' or 'n' POS tags; this should not be too important,
							# since those nonsense hyponyms taking other POS tags would not find themselves somewhere in the corpus
							# anyway, so they are filtered out by the global_presence filter.
							if vnonly and cur_lex_ent['pos'] not in ['v', 'n']:
								continue
							for sense_id, cur_sense in enumerate(cur_lex_ent['senses']):
								if sense_id > 0 and firstonly:
									continue
								cur_sst = synsets[cur_sense]
								related_ssts = []
								for rtp in accepted_neg_indicator_reltypes:
									if rtp in cur_sst['rels']:
										related_ssts += cur_sst['rels'][rtp]
								related_ssts = list(set(related_ssts))
								for rsst in related_ssts:
									if rsst not in sstids2lemmas:
										continue
									else:
										cur_hsyn_all_predstrs = []
										for hlem in sstids2lemmas[rsst]:
											# if hlem == cur_span:
											# 	print("!!!")
											cur_hsyn_all_predstrs.append(prefix+hlem+postfix)
										for hpredstr in cur_hsyn_all_predstrs:
											if hpredstr not in potneg_replacements:
												potneg_replacements[hpredstr] = set()
											potneg_replacements[hpredstr].update(cur_hsyn_all_predstrs)
					elif lang == 'en':
						potneg_replacements = dict()
						# if all tokens are prepositions, then there is no point in replacing it.
						if not all_toks_are_preps(upred_list, start_id, end_id):
							span_to_replace = '_'.join(upred_list[start_id:end_id])
							if tok_synsets is not None and span_to_replace in tok_synsets:
								try:
									syns = [wordnet.synset_from_sense_key(x) for x in tok_synsets[span_to_replace]]
								except Exception as e:
									if not str(e).startswith('lemma') and not str(e).startswith('adjective'):
										print(e, file=sys.stderr)
									syns = None
							else:
								syns = None

							if syns is None:
								syns = wordnet.synsets(span_to_replace)
							elif len(syns) == 0 and allow_backoff_wsd:
								syns = wordnet.synsets(span_to_replace)

							for syn in syns:
								span_hyponyms = syn.hyponyms() + syn.instance_hyponyms()
								"""
								``entailments'' are where the positive is on the premise side of a relation, but we
								want the positives to be on the hypothesis side in comparison to the negatives.
								"""
								# span_entailments = syn.entailments()
								"""
								``causes'' are mostly synsets of the same lemma and only different in indices; 
								when the lemmas are different, still it looks like unclear which entails which, 
								looks very much like symmetric paraphrases.
								Therefore, I am not using this.
								"""
								# span_causes = syn.causes()
								# if len(span_causes) > 0:
								# 	print(f"posi_synset: {syn}; span_causes: {span_causes}")
								for hsyn in span_hyponyms:
									hypo_lemm_overlap_with_posi_flag = False
									cur_hsyn_all_predstrs = []
									h_lemmas = hsyn.lemma_names()
									for hlem in h_lemmas:
										if hlem == span_to_replace:
											hypo_lemm_overlap_with_posi_flag = True
											break
										hlem = hlem.replace('_', ' ')
										cur_hyponym = ' '.join(upred_list[:start_id] + [hlem] + upred_list[end_id:])
										cur_hsyn_all_predstrs.append(cur_hyponym)

									# all lemmas in the hypothesis synset must not be the same as the original span to replace.
									if hypo_lemm_overlap_with_posi_flag:
										continue

									for hpredstr in cur_hsyn_all_predstrs:
										if hpredstr not in potneg_replacements:
											potneg_replacements[hpredstr] = set()
										potneg_replacements[hpredstr].update(cur_hsyn_all_predstrs)
								if firstonly:
									break

					else:
						raise AssertionError

					potneg_triple_strs = {f"{x}::{subj}::{obj}::{tsubj}::{tobj}": lst for (x, lst) in
										  potneg_replacements.items()}

					# ATTENTION: since now we process only the entries in a single partition per round, we do not need to
					# load triples for each posi_entry any more!
					#
					# if posi_entry['partition_key'] not in partition_triple_strings:
					# 	# the ``cur_partition_preds'' has been unused.
					# 	cur_partition_triples, cur_partition_preds = load_triple_set(partition_triples_path % posi_entry['partition_key'], verbose=False)
					# 	# by setting threshold to 0, this line below simply returns the set of tplstrs
					# 	# (since we don't need the actual triples, this would be more convenient)
					# 	cur_tplstrs2exclude = filter_sound_triples(cur_partition_triples, 0)
					# 	assert len(cur_partition_triples) == 0
					# 	del cur_partition_triples
					# 	partition_triple_strings[posi_entry['partition_key']] = cur_tplstrs2exclude
					# else:
					# 	cur_tplstrs2exclude = partition_triple_strings[posi_entry['partition_key']]
					# assert isinstance(cur_tplstrs2exclude, set)

					cur_span_is_good_flag = False
					# print(potneg_triple_strs)
					# time.sleep(8)
					for tplstr in potneg_triple_strs:
						# assert tplstr != posi_tplstr, f"tplstr: {tplstr}"
						negtpl_list = tplstr.split('::')
						assert len(negtpl_list) == 5  # pred::subj::obj::tsubj::tobj

						# this_tplstr_matched_in_lst_flag = False

						# a negative sample must not have occurred in the current partition, or is specified so, not in
						# any partition.
						if tplstr in cur_tplstrs2exclude:
							continue
						elif tplstr == posi_tplstr:
							print(f"tplstr: Unexpected Behavior!")
							print(f"posi_tplstr: {posi_tplstr}")
							continue

						equiv_exists = False

						# the other lemma forms in the same synset as a negative sample must not have occurred in the
						# current partition, and if specified so, not in any partition. (namely banning synonyms)
						for equiv_neg_predstr in potneg_triple_strs[tplstr]:
							equiv_neg_tplstr = '::'.join([equiv_neg_predstr]+negtpl_list[1:])
							# if equiv_neg_tplstr == tplstr:
							# 	this_tplstr_matched_in_lst_flag = True
							# time.sleep(8)
							# print(f"tplstr: {tplstr}")
							# print(f"equiv_neg_tplstr: {equiv_neg_tplstr}")
							# print(f"equiv_neg_predstr: {equiv_neg_predstr}")
							if equiv_neg_tplstr in cur_tplstrs2exclude:
								equiv_exists = True
								continue
							elif equiv_neg_tplstr == posi_tplstr:
								print(f"equiv_neg_tplstr: Unexpected Behavior!")
								print(f"posi_tplstr: {posi_tplstr}")
								assert posi_tplstr not in cur_tplstrs2exclude
								equiv_exists = True
								continue
							else:
								pass

						if equiv_exists:
							continue

						# assert this_tplstr_matched_in_lst_flag is True, f"tplstr: {tplstr}"

						neg_predstr = negtpl_list[0]
						# print(f"neg predstr: {neg_predstr}; pos predstr: {posi_predstr}")
						# time.sleep(8)
						assert neg_predstr != posi_predstr, f"neg predstr: {neg_predstr}; pos predstr: {posi_predstr}"
						neg_key_predstr = None
						lemm_neg_predstr, lemm_noprep_neg_predstr = lemmatize_str(neg_predstr, lemmatizer, lang=lang)
						lemm_neg_key_predstr = None

						# but a challenging negative sample must have occurred in some partition of the corpus.
						if global_presence == 'triple':
							if tplstr not in sound_tplstrs:
								continue
						elif global_presence == 'pred':
							if neg_predstr in all_preds[type_pair] and \
									fits_thresholds(all_preds[type_pair][neg_predstr]['n'], global_presence_thres,
													global_presence_cap):
								neg_key_predstr = neg_predstr
								lemm_neg_key_predstr = lemm_neg_predstr
								pass
							else:
								approx_found = False
								assert neg_key_predstr is None
								if lemm_neg_predstr in lemm_2_predstr[type_pair]:
									for approx_neg_predstr in lemm_2_predstr[type_pair][lemm_neg_predstr]:
										if approx_neg_predstr in all_preds[type_pair] and approx_neg_predstr != posi_predstr and \
												fits_thresholds(all_preds[type_pair][approx_neg_predstr]['n'],
																global_presence_thres, global_presence_cap):
											if neg_key_predstr is None:
												assert lemm_neg_key_predstr is None
												neg_key_predstr = approx_neg_predstr
												lemm_neg_key_predstr = lemm_neg_predstr
											approx_found = True
								if approx_found is True:

									pass
								else:
									assert approx_found is False and neg_key_predstr is None
									if allow_prep_mismatch_replacements and len(
											lemm_noprep_neg_predstr) > 0 and lemm_noprep_neg_predstr in \
											lemm_noprep_2_predstr[type_pair]:
										for approx_neg_predstr in lemm_noprep_2_predstr[type_pair][
											lemm_noprep_neg_predstr]:
											if approx_neg_predstr in all_preds[type_pair] and approx_neg_predstr != posi_predstr and \
													fits_thresholds(all_preds[type_pair][approx_neg_predstr]['n'],
																	global_presence_thres, global_presence_cap):
												if neg_key_predstr is None:
													assert lemm_neg_key_predstr is None
													neg_key_predstr = approx_neg_predstr
													lemm_approx_neg_predstr, _ = lemmatize_str(approx_neg_predstr,
																							   lemmatizer, lang=lang)
													lemm_neg_key_predstr = lemm_approx_neg_predstr
												approx_found = True
									if approx_found is True:
										pass
									else:
										# in this case we are sure not felicitous negative predicates can be constructed
										# for this #potneg_predstr#, so we abandon this one and continue to the next.
										continue
						else:
							raise AssertionError

						# the recorded curneg_upred is the most frequent upred form of this upred_str
						assert neg_key_predstr is not None and lemm_neg_key_predstr is not None

						# print(f"neg key predstr: {neg_key_predstr}; pos predstr: {posi_predstr}")
						# time.sleep(8)
						assert neg_key_predstr != posi_predstr, f"neg key predstr: {neg_key_predstr}; pos predstr: {posi_predstr}"

						curneg_upred = all_preds[type_pair][neg_key_predstr]['p'][0][0]
						assert curneg_upred != upred
						# print(f"pos_upred: {upred}; good_neg_upred: {curneg_upred}")
						neg_predstrs_after_filter.add(neg_key_predstr)
						curneg_rel = assemble_rel(curneg_upred, posi_entry)
						curneg_entry = copy.deepcopy(posi_entry)
						curneg_entry['r'] = curneg_rel
						curneg_entry['neg_predstr'] = neg_key_predstr
						curneg_entry['lemm_neg_predstr'] = lemm_neg_key_predstr
						curneg_entry['posi_upred'] = upred
						curneg_entry['posi_idx'] = posi_idx
						curneg_entry['type'] = 'wn'
						potential_negatives.append(curneg_entry)
						cur_span_is_good_flag = True

					if cur_span_is_good_flag:
						negfound_predstr_spans.append({'start_id': start_id, 'end_id': end_id})

			posi_entry['negatives'] = list(neg_predstrs_after_filter)

			if len(posi_entry['negatives']) > 0:
				negfound_sidxes.add(posi_entry['in_partition_sidx'])

			num_negs_found.append(len(neg_predstrs_after_filter))

	all_posi_idxes = set(range(len(positives)))
	# assert len(all_posi_idxes.difference(processed_posi_idxes)) == 0

	avg_num_negs = sum(num_negs_found) / len(num_negs_found) if len(num_negs_found) > 0 else 0.0
	count_some_negs_found = len([x for x in num_negs_found if x > 0])
	count_five_negs_found = len([x for x in num_negs_found if x >= 5])
	count_ten_negs_found = len([x for x in num_negs_found if x >= 10])

	print(f"Total number of negatives: {sum(num_negs_found)}")
	print(f"Average number of negatives: {avg_num_negs}")
	print(f"Count all positives: {len(num_negs_found)}")
	print(f"Count some negs found: {count_some_negs_found}")
	print(f"Count >=5 negs found: {count_five_negs_found}")
	print(f"Count >=10 negs found: {count_ten_negs_found}")

	return potential_negatives, num_negs_found


def find_negatives_word2vec(positives, h5file, mismatches_dset, word_vectors, global_tplstrs, sound_tplstrs, all_preds,
							partition_triples_path,
							global_presence, global_presence_thres, global_presence_cap, global_triple_absence_flag,
							similarity_thres, similarity_cap, ranking_thres,
							ranking_cap, negs_per_pos, max_population_size, max_num_posi_collected_per_partition,
							pred_vectors_cache_size, verbose, lang):
	lemmatizer = WordNetLemmatizer()

	assert global_presence in ['triple', 'pred']
	stopwords = ['X', '的', '【介宾】']

	pred_vectors = h5file['upred_vecs']
	print(h5file.keys())

	potential_negatives_extensions = []
	potential_negatives_non_extensions = []
	num_negs_found_extensions = []
	num_negs_found_non_extensions = []

	all_partition_keys = set()
	for posi_entry in positives:
		all_partition_keys.add(posi_entry['partition_key'])
	all_partition_keys = list(all_partition_keys)
	all_partition_keys.sort()

	processed_posi_idxes = set()
	print(f"Beginning to process.")
	st = time.time()
	dur_loadfile = 0.0
	# dur_shuffling = 0.0
	dur_cosine = 0.0
	# dur_posivec = 0.0
	dur_loadnegivec = 0.0
	# dur_judgeext = 0.0
	# dur_subsample = 0.0
	dur_fetchcurneg = 0.0
	dur_fetchbuffneg = 0.0

	predvecs_cache = {}
	cache_hits = 0.0
	cache_queries = 0.0

	for pk in all_partition_keys:
		st_loadfile = time.time()
		if global_triple_absence_flag:
			cur_tplstrs2exclude = global_tplstrs
		else:
			cur_partition_triples, cur_partition_preds = load_triple_set(partition_triples_path % pk, verbose=False,
																		 lang=lang)
			cur_tplstrs2exclude = set(cur_partition_triples.keys())
			assert isinstance(cur_tplstrs2exclude, set)
			del cur_partition_triples
			del cur_partition_preds
		print(f"fetching negatives for positives in partition: {pk}")
		et_loadfile = time.time()
		dur_loadfile += (et_loadfile - st_loadfile)

		# below is the set of sentence indices where some negatives can be found from some of the positives in this sentence.
		# the size of this set is the same as the number of positives attended in this partition, since we are going to
		# dictate that only one positive per sentence can be selected into the pool.
		negfound_sidxes = set()
		all_posi_idxes = list(range(len(positives)))
		random.shuffle(all_posi_idxes)
		cur_partition_num_posis_processed = 0

		# this flag is set because we have multiple iterations where cur_partition_num_posis_processed doesn't change,
		# but we don't want to print the same thing out multiple times: this is set to False after each printing, and
		# reset to True after ``cur_partition_num_posis_processed += 1''
		print_flag = True

		for posi_idx in all_posi_idxes:
			posi_entry = positives[posi_idx]

			if cur_partition_num_posis_processed % 100 == 1 and print_flag is True:
				ct = time.time()
				dur = ct - st
				duration_format_print(dur, f"cur_partition_num_posis_processed: {cur_partition_num_posis_processed}")
				# duration_format_print(dur_shuffling, f"dur_shuffling")
				duration_format_print(dur_cosine, f"dur_cosine")
				# duration_format_print(dur_posivec, f"dur_posivec")
				duration_format_print(dur_loadnegivec, f"dur_loadnegivec")
				# duration_format_print(dur_judgeext, f"dur_judgeext")
				# duration_format_print(dur_subsample, f"dur_subsample")
				duration_format_print(dur_loadfile, f"dur_loadfile")
				duration_format_print(dur_fetchcurneg, f"dur_fetchcurneg")
				duration_format_print(dur_fetchbuffneg, f"dur_fetchbuffneg")
				print(f"predvec memory cache size: %d; hit rate: %.2f percents" % (
					len(predvecs_cache), 100 * cache_hits / cache_queries))
				print("")
				cache_hits = 0.0
				cache_queries = 0.0
				print_flag = False

			if posi_entry['partition_key'] != pk:
				continue
			cur_partition_num_posis_processed += 1
			print_flag = True

			# if there has been other positives selected from this sentence, then no other positives are considered.
			if posi_entry['in_partition_sidx'] in negfound_sidxes:
				continue

			if len(negfound_sidxes) > max_num_posi_collected_per_partition:
				print(f"Maximum number of positives to collect reached for current partition: ``{pk}''")
				break

			assert posi_idx not in processed_posi_idxes
			processed_posi_idxes.add(posi_idx)

			posi_upred, subj, obj, tsubj, tobj = parse_rel(posi_entry)
			posi_tplstr, posi_predstr = rel2concise_str(posi_upred, subj, obj, tsubj, tobj, lang=lang)
			upred_list = upred2bow(posi_upred, lang=lang)

			if check_vague(upred_list) is True:
				continue

			# st_posivec = time.time()

			upred_vecs = []
			type_pair = f"{tsubj}#{tobj}"

			posi_mismatch = False
			for utok in upred_list:
				if utok in stopwords:
					continue
				if utok not in word_vectors:
					if lang in ['zh', 'en']:
						posi_mismatch = True
						break
					else:
						raise AssertionError
				upred_vecs.append(word_vectors[utok])

			if posi_mismatch or len(upred_vecs) == 0:
				num_negs_found_extensions.append(0)
				num_negs_found_non_extensions.append(0)
				continue

			upred_mean_vec = np.mean(upred_vecs, axis=0)

			# et_posivec = time.time()
			# dur_posivec += (et_posivec - st_posivec)

			# unlike in WordNet setting, here this does not have to be a set, because the keys in the ``all_preds'' below are already unique
			neg_upredstrs_after_filter_extensions = []
			neg_upredstrs_after_filter_non_extensions = []
			cur_population_size = 0

			# ATTENTION: since now we process only the entries in a single partition per round, we do not need to
			# load triples for each posi_entry any more!
			#
			# if posi_entry['partition_key'] not in partition_triple_strings:
			# 	# the ``cur_partition_preds'' has been unused
			# 	cur_partition_triples, cur_partition_preds = load_triple_set(partition_triples_path % posi_entry['partition_key'],
			# 											   verbose=False)
			# 	# by setting threshold to 0, this line below simply returns the set of tplstrs
			# 	# (since we don't need the actual triples, this would be more convenient)
			# 	cur_tplstrs2exclude = filter_sound_triples(cur_partition_triples, 0)
			# 	partition_triple_strings[posi_entry['partition_key']] = cur_tplstrs2exclude
			# 	del cur_partition_triples
			# else:
			# 	cur_tplstrs2exclude = partition_triple_strings[posi_entry['partition_key']]

			# st_shuffling = time.time()
			negpredstr_population = list(all_preds[type_pair].keys())
			# we do not need to shuffle the whole thing every time! just selecting a random start point would suffice!
			this_startpoint = random.randrange(len(negpredstr_population))
			negpredstr_population = negpredstr_population[this_startpoint:] + negpredstr_population[:this_startpoint]
			# random.shuffle(negpredstr_population)
			# et_shuffling = time.time()
			# dur_shuffling += (et_shuffling - st_shuffling)

			# these buffers are set out for batch processing of cosine similarity.
			neg_predstr_buffer = []
			neg_predvec_buffer = []
			max_popsize_reached_flag = False

			for curneg_i, curneg_predstr in enumerate(negpredstr_population):
				# This check has already been done in the two wrappers! (predicates in the mismatches_dset have already been filtered out)
				# if curneg_predstr in mismatches_dset:
				# 	continue
				if curneg_predstr == posi_predstr:
					continue
				if max_popsize_reached_flag:
					break

				st_fetchcurneg = time.time()
				curneg_global_id = all_preds[type_pair][curneg_predstr]['global_id']
				# for Word2Vec setting, the predicate is guaranteed to be present in some partition of the corpus,
				# so it's just a matter of the number of occurrences.
				curneg_num_occurrences = all_preds[type_pair][curneg_predstr]['n']
				et_fetchcurneg = time.time()
				dur_fetchcurneg += (et_fetchcurneg - st_fetchcurneg)

				curneg_occured_in_current_partition = False
				for curneg_pot_upred, _ in all_preds[type_pair][curneg_predstr]['p']:
					# a negative sample must not have occurred in the current partition;
					curneg_triplestr = f"{curneg_pot_upred}::{subj}::{obj}::{tsubj}::{tobj}"
					if curneg_triplestr in cur_tplstrs2exclude:
						curneg_occured_in_current_partition = True
						break
					elif curneg_triplestr == posi_tplstr:
						print(f"tplstr: Unexpected Behavior!")
						print(f"posi_tplstr: {posi_tplstr}")
						continue
					else:
						pass

					# TODO: Remember to also check the synonyms.
				if curneg_occured_in_current_partition:
					continue

				# but a challenging negative sample must have occurred in some partition of the corpus.
				# TODO: the global_presence_thres should be tuned according to whether presence is defined by triples or predicates (higher bars for predicates!)
				if global_presence == 'triple':
					curneg_best_upred = all_preds[type_pair][curneg_predstr]['p'][0][0]
					curneg_best_triplestr = f"{curneg_best_upred}::{subj}::{obj}::{tsubj}::{tobj}"
					if curneg_best_triplestr not in sound_tplstrs:
						continue
				elif global_presence == 'pred':
					if not fits_thresholds(curneg_num_occurrences, global_presence_thres, global_presence_cap):
						continue
				else:
					raise AssertionError

				st_loadnegivec = time.time()
				if curneg_global_id in predvecs_cache:
					curneg_vec = predvecs_cache[curneg_global_id]
					cache_hits += 1
				else:
					curneg_vec = pred_vectors[curneg_global_id]
					if len(predvecs_cache) < pred_vectors_cache_size:
						predvecs_cache[curneg_global_id] = curneg_vec
				cache_queries += 1
				et_loadnegivec = time.time()
				dur_loadnegivec += (et_loadnegivec - st_loadnegivec)

				neg_predstr_buffer.append(curneg_predstr)
				neg_predvec_buffer.append(curneg_vec)

				if len(neg_predstr_buffer) >= 1000 or curneg_i == len(negpredstr_population) - 1:
					assert len(neg_predstr_buffer) == len(neg_predvec_buffer)
					st_cosine = time.time()
					neg_predvec_buffer = np.array(neg_predvec_buffer)
					buffneg_simscores = calc_simscore(upred_mean_vec.reshape(1, -1), neg_predvec_buffer)
					et_cosine = time.time()
					dur_cosine += (et_cosine - st_cosine)
					assert buffneg_simscores.shape[0] == 1
					assert buffneg_simscores.shape[1] == len(neg_predstr_buffer)

					for buff_id in range(len(neg_predstr_buffer)):
						st_fetchbuffneg = time.time()
						buffneg_predstr = neg_predstr_buffer[buff_id]
						buffneg_simscore = buffneg_simscores[0, buff_id]
						buffneg_upred = all_preds[type_pair][buffneg_predstr]['p'][0][0]
						et_fetchbuffneg = time.time()
						dur_fetchbuffneg += (et_fetchbuffneg - st_fetchbuffneg)

						# apart from ranking, additionally use a similarity hard threshold.
						if buffneg_simscore < similarity_thres:
							continue
						elif buffneg_simscore > similarity_cap:
							if verbose:
								print(
									f"Too similar negative exceeding similarity cap of {similarity_cap}: {buffneg_upred}; {buffneg_simscore};")
							continue

						# if math.isnan(buffneg_simscore):
						# 	print(f"posi_predstr: {posi_predstr}", file=sys.stderr)
						# 	print(f"buffneg_predstr: {buffneg_predstr}", file=sys.stderr)
						# 	print(f"posi vec: {upred_mean_vec.reshape(1, -1)}", file=sys.stderr)
						# 	print(f"buffneg vec: {neg_predvec_buffer[buff_id, :].reshape(1, -1)}", file=sys.stderr)

						# st_judgeext = time.time()
						buffneg_upred_list = upred2bow(buffneg_upred, lang=lang)
						all_posi_toks_in_buffneg = True
						for posi_tok in upred_list:
							this_posi_tok_in_buffnegtok = False
							for negi_tok in buffneg_upred_list:
								if posi_tok in negi_tok:
									this_posi_tok_in_buffnegtok = True
									break
							if not this_posi_tok_in_buffnegtok:
								all_posi_toks_in_buffneg = False
								break
						# et_judgeext = time.time()
						# dur_judgeext += (et_judgeext - st_judgeext)

						if all_posi_toks_in_buffneg:
							neg_upredstrs_after_filter_extensions.append((buffneg_predstr, buffneg_simscore))
						else:
							neg_upredstrs_after_filter_non_extensions.append((buffneg_predstr, buffneg_simscore))
						cur_population_size += 1
						if cur_population_size > max_population_size:
							print(
								f"Maximum population size reached at the {cur_partition_num_posis_processed} 'th positive processed, after looking at {curneg_i} potential negatives!")
							max_popsize_reached_flag = True
							break
					# if len(neg_upreds_after_filter) > 10000:  # periodically clear the cache, to keep memory occupation managable, while not increasing complexity much.
					# 	neg_upreds_after_filter = neg_upreds_after_filter.sort(key=lambda x: x[1], reverse=True)
					# 	neg_upreds_after_filter = neg_upreds_after_filter[:ranking_thres]

					neg_predstr_buffer = []
					neg_predvec_buffer = []

			# st_subsample = time.time()
			# sort the population of ``challenging negatives'' whose similarity is not exactly one, in descending order of similarity.
			neg_upredstrs_after_filter_extensions.sort(key=lambda x: x[1], reverse=True)
			neg_upredstrs_after_filter_non_extensions.sort(key=lambda x: x[1], reverse=True)
			neg_upredstrs_after_filter_extensions = [x for x in neg_upredstrs_after_filter_extensions if x[1] < 1.]
			neg_upredstrs_after_filter_non_extensions = [x for x in neg_upredstrs_after_filter_non_extensions if
														 x[1] < 1.]
			if verbose:
				print(f"For the positive predicate: {posi_upred}: ")
				print(
					f"Too similar negatives excluded by ranking cap (extensions): {neg_upredstrs_after_filter_extensions[:ranking_cap]}")
				print(
					f"Too similar negatives excluded by ranking cap (non-extensions): {neg_upredstrs_after_filter_non_extensions[:ranking_cap]}")

			# sample ``sample_per_pos'' negatives from the top [ranking_cap:ranking_thres] challenging negatives that are most similar to the positive
			assert ranking_cap < ranking_thres
			neg_upredstrs_after_filter_extensions = neg_upredstrs_after_filter_extensions[ranking_cap:ranking_thres]
			neg_upredstrs_after_filter_non_extensions = neg_upredstrs_after_filter_non_extensions[
														ranking_cap:ranking_thres]
			neg_upredstrs_after_filter_extensions = random.sample(neg_upredstrs_after_filter_extensions,
																  k=negs_per_pos) if len(
				neg_upredstrs_after_filter_extensions) > negs_per_pos else neg_upredstrs_after_filter_extensions
			neg_upredstrs_after_filter_non_extensions = random.sample(neg_upredstrs_after_filter_non_extensions,
																	  k=negs_per_pos) if len(
				neg_upredstrs_after_filter_non_extensions) > negs_per_pos else neg_upredstrs_after_filter_non_extensions

			for curneg_predstr in neg_upredstrs_after_filter_extensions:
				if verbose:
					print(
						f"[EXT] pos_upred: {posi_upred}; good_neg_predstr: {curneg_predstr[0]}; simscore: {curneg_predstr[1]}")
				curneg_upred = all_preds[type_pair][curneg_predstr]['p'][0][
					0]  # we select the most frequent upred associated with the current neg_predstr
				lemm_curneg_predstr, _ = lemmatize_str(curneg_predstr, lemmatizer, lang=lang)
				curneg_rel = assemble_rel(curneg_upred, posi_entry)
				curneg_entry = copy.deepcopy(posi_entry)
				curneg_entry['r'] = curneg_rel
				curneg_entry['posi_upred'] = posi_upred
				curneg_entry['posi_idx'] = posi_idx
				curneg_entry['neg_predstr'] = curneg_predstr
				curneg_entry['lemm_neg_predstr'] = lemm_curneg_predstr
				curneg_entry['type'] = 'ext'
				potential_negatives_extensions.append(curneg_entry)
			for curneg_predstr in neg_upredstrs_after_filter_non_extensions:
				if verbose:
					print(
						f"[NONEXT] pos_upred: {posi_upred}; good_neg_upred: {curneg_predstr[0]}; simscore: {curneg_predstr[1]}")
				curneg_upred = all_preds[type_pair][curneg_predstr]['p'][0][
					0]  # we select the most frequent upred associated with the current neg_predstr
				lemm_curneg_predstr, _ = lemmatize_str(curneg_predstr, lemmatizer, lang=lang)
				curneg_rel = assemble_rel(curneg_upred, posi_entry)
				curneg_entry = copy.deepcopy(posi_entry)
				curneg_entry['r'] = curneg_rel
				curneg_entry['posi_upred'] = posi_upred
				curneg_entry['posi_idx'] = posi_idx
				curneg_entry['neg_predstr'] = curneg_predstr
				curneg_entry['lemm_neg_predstr'] = lemm_curneg_predstr
				curneg_entry['type'] = 'non-ext'
				potential_negatives_non_extensions.append(curneg_entry)

			posi_entry['negatives'] = neg_upredstrs_after_filter_extensions + neg_upredstrs_after_filter_non_extensions
			# et_subsample = time.time()
			# dur_subsample += (et_subsample - st_subsample)

			# if there are negatives found for this positive, count its sidx in to the set of ``negfound_sidxes''
			if len(posi_entry['negatives']) > 0:
				negfound_sidxes.add(posi_entry['in_partition_sidx'])

			num_negs_found_extensions.append(len(neg_upredstrs_after_filter_extensions))
			num_negs_found_non_extensions.append(len(neg_upredstrs_after_filter_non_extensions))

	# TODO: others: re-do the construction of all_triples and all_preds, such that ``global_ids'' and ``num_occ'' are included

	all_posi_idxes = set(range(len(positives)))
	# assert len(all_posi_idxes.difference(processed_posi_idxes)) == 0

	avg_num_negs_extensions = sum(num_negs_found_extensions) / len(num_negs_found_extensions) if len(
		num_negs_found_extensions) > 0 else 0.0
	avg_num_negs_non_extensions = sum(num_negs_found_non_extensions) / len(num_negs_found_non_extensions) if len(
		num_negs_found_non_extensions) > 0 else 0.0
	print(f"Total number of negatives (by extensions): {sum(num_negs_found_extensions)}")
	print(f"Average number of negatives (by extensions): {avg_num_negs_extensions}")
	print(f"Total number of negatives (not by extensions): {sum(num_negs_found_non_extensions)}")
	print(f"Average number of negatives (not by extensions): {avg_num_negs_non_extensions}")

	return potential_negatives_extensions, num_negs_found_extensions, potential_negatives_non_extensions, num_negs_found_non_extensions


def find_negatives_wrapper(args, wordnet_dict, h5file, word_vectors, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
						   lemm_noprep_2_predstr, do_which):
	dev_positives = []
	test_positives = []
	with open(args.positives_dev_fn, 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			dev_positives.append(item)
	with open(args.positives_test_fn, 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			test_positives.append(item)

	if do_which == 'wordnet':
		assert wordnet_dict is not None or args.lang == 'en'
		dev_negatives, _ = find_negatives_wordnet(dev_positives, wordnet_dict, global_tplstrs, sound_tplstrs, all_preds,
												  lemm_2_predstr, lemm_noprep_2_predstr,
												  args.partition_triples_path,
												  args.vnonly, args.firstonly, args.global_presence,
												  args.global_presence_thres, args.global_presence_cap,
												  args.global_triple_absence_flag,
												  args.max_num_posi_collected_per_partition, args.verbose, args.lang,
												  args.allow_prep_mismatch_replacements, args.allow_backoff_wsd)
		test_negatives, _ = find_negatives_wordnet(test_positives, wordnet_dict, global_tplstrs, sound_tplstrs, all_preds,
												   lemm_2_predstr, lemm_noprep_2_predstr,
												   args.partition_triples_path,
												   args.vnonly, args.firstonly, args.global_presence,
												   args.global_presence_thres, args.global_presence_cap,
												   args.global_triple_absence_flag,
												   args.max_num_posi_collected_per_partition, args.verbose, args.lang,
												   args.allow_prep_mismatch_replacements, args.allow_backoff_wsd)
	elif do_which == 'word2vec':
		assert h5file is not None
		assert word_vectors is not None
		mismatches_dset = set([x.decode('utf-8') for x in h5file['mismatched_predstrs']])
		print(f"size of mismatches_dset: {len(mismatches_dset)}")
		for type_pair in all_preds:
			this_keys = list(all_preds[type_pair].keys())
			for k in this_keys:
				if k in mismatches_dset:
					del all_preds[type_pair][k]
		print(f"Deleted mismatched predicates from ``all_preds''!")
		dev_negatives_ext, _, dev_negatives_nonext, _ = find_negatives_word2vec(dev_positives, h5file, mismatches_dset,
																				word_vectors, global_tplstrs, sound_tplstrs, all_preds,
																				args.partition_triples_path,
																				args.global_presence,
																				args.global_presence_thres,
																				args.global_presence_cap,
																				args.global_triple_absence_flag,
																				args.similarity_thres,
																				args.similarity_cap,
																				args.ranking_thres, args.ranking_cap,
																				args.negs_per_pos, args.population_size,
																				args.max_num_posi_collected_per_partition,
																				args.pred_vectors_cache_size,
																				args.verbose, lang=args.lang)
		test_negatives_ext, _, test_negatives_nonext, _ = find_negatives_word2vec(test_positives, h5file,
																				  mismatches_dset, word_vectors,
																				  global_tplstrs, sound_tplstrs, all_preds,
																				  args.partition_triples_path,
																				  args.global_presence,
																				  args.global_presence_thres,
																				  args.global_presence_cap,
																				  args.global_triple_absence_flag,
																				  args.similarity_thres,
																				  args.similarity_cap,
																				  args.ranking_thres, args.ranking_cap,
																				  args.negs_per_pos,
																				  args.population_size,
																				  args.max_num_posi_collected_per_partition,
																				  args.pred_vectors_cache_size,
																				  args.verbose, lang=args.lang)
		dev_negatives = truncate_merge_by_ratio(dev_negatives_ext, dev_negatives_nonext, args.ext_ratio,
												1 - args.ext_ratio)
		test_negatives = truncate_merge_by_ratio(test_negatives_ext, test_negatives_nonext, args.ext_ratio,
												 1 - args.ext_ratio)

	else:
		raise AssertionError

	with open(args.negatives_dev_fn % do_which, 'w', encoding='utf8') as fp:
		for item in dev_negatives:
			out_line = json.dumps(item, ensure_ascii=False)
			fp.write(out_line + '\n')
	with open(args.negatives_test_fn % do_which, 'w', encoding='utf8') as fp:
		for item in test_negatives:
			out_line = json.dumps(item, ensure_ascii=False)
			fp.write(out_line + '\n')
	return


def find_potential_negatives(args, wordnet_dict, h5file, word_vectors, global_tplstrs: set, sound_tplstrs, all_preds, lemm_2_predstr,
							 lemm_noprep_2_predstr, time_slices, do_which, single_pot_partition_idx=None):
	assert global_tplstrs is None or isinstance(global_tplstrs, set)
	all_potential_negatives = []
	all_num_neg_found = []
	if single_pot_partition_idx is not None:
		if single_pot_partition_idx >= len(time_slices):
			print(
				f"Single partition potentials: partition index {single_pot_partition_idx} out of range for time slices with size {len(time_slices)}!")
			return
		else:
			print(
				f"Fetching negatives only for potential positives in partition: {time_slices[single_pot_partition_idx]}!")

	if do_which == 'word2vec':
		mismatches_dset = set([x.decode('utf-8') for x in h5file['mismatched_predstrs']])
		print(f"size of mismatches_dset: {len(mismatches_dset)}")
		for type_pair in all_preds:
			this_keys = list(all_preds[type_pair].keys())
			for k in this_keys:
				if k in mismatches_dset:
					del all_preds[type_pair][k]
		print(f"Deleted mismatched predicates from ``all_preds''!")
	elif do_which == 'wordnet':
		pass
	else:
		raise AssertionError

	for t_slice_idx, t_slice in enumerate(time_slices):
		if single_pot_partition_idx is not None and t_slice_idx != single_pot_partition_idx:
			continue

		print(f"Fetching negatives for potential positives in {t_slice}")
		cur_positives = []

		with open(args.potential_positives_path % t_slice, 'r', encoding='utf8') as in_fp:
			for line in in_fp:
				item = json.loads(line)
				cur_positives.append(item)
		print(f"Positives loaded.")
		if do_which == 'wordnet':
			assert wordnet_dict is not None or args.lang == 'en'
			cur_negatives, cur_num_neg_found = find_negatives_wordnet(cur_positives, wordnet_dict, global_tplstrs, sound_tplstrs,
																	  all_preds, lemm_2_predstr, lemm_noprep_2_predstr,
																	  args.partition_triples_path, args.vnonly,
																	  args.firstonly, args.global_presence,
																	  args.global_presence_thres, args.global_presence_cap,
																	  args.global_triple_absence_flag,
																	  args.max_num_posi_collected_per_partition,
																	  args.verbose,
																	  args.lang, args.allow_prep_mismatch_replacements,
																	  args.allow_backoff_wsd)
		elif do_which == 'word2vec':
			assert h5file is not None
			assert word_vectors is not None
			cur_negatives_ext, cur_num_neg_found_ext, cur_negatives_nonext, cur_num_neg_found_nonext = find_negatives_word2vec(
				cur_positives, h5file, mismatches_dset, word_vectors, global_tplstrs, sound_tplstrs, all_preds,
				args.partition_triples_path, args.global_presence,
				args.global_presence_thres, args.global_presence_cap, args.global_triple_absence_flag,
				args.similarity_thres, args.similarity_cap,
				args.ranking_thres, args.ranking_cap, args.negs_per_pos,
				args.population_size, args.max_num_posi_collected_per_partition,
				args.pred_vectors_cache_size, args.verbose, lang=args.lang)
			cur_negatives = truncate_merge_by_ratio(cur_negatives_ext, cur_negatives_nonext, args.ext_ratio,
													1 - args.ext_ratio)
			if len(cur_num_neg_found_ext) != len(cur_num_neg_found_nonext):
				print(
					f"ext and nonext length not same! ext: {len(cur_num_neg_found_ext)}; non-ext: {len(cur_num_neg_found_nonext)}")
			cur_num_neg_found = [len(cur_negatives) / len(cur_num_neg_found_ext) for x in cur_num_neg_found_ext]
		else:
			raise AssertionError
		print("Dumping retrieved negatives...")
		with open(args.potential_negatives_path % (do_which, t_slice), 'w', encoding='utf8') as out_fp:
			for item in cur_negatives:
				item_neg_upred, _, _, _, _ = parse_rel(item)
				assert item_neg_upred != item['posi_upred'], print(f"item: {item}")
				out_line = json.dumps(item, ensure_ascii=False)
				out_fp.write(out_line + '\n')
		all_potential_negatives += cur_negatives
		all_num_neg_found += cur_num_neg_found

	print(f"Dumping all retrieved negatives...")
	with open(args.potential_negatives_path % (do_which, 'all'), 'w', encoding='utf8') as out_fp:
		for item in all_potential_negatives:
			out_line = json.dumps(item, ensure_ascii=False)
			out_fp.write(out_line + '\n')

	all_avg_num_negs = sum(all_num_neg_found) / len(all_num_neg_found) if len(all_num_neg_found) > 0 else 0.0
	all_count_some_negs_found = len([x for x in all_num_neg_found if x > 0])
	all_count_five_negs_found = len([x for x in all_num_neg_found if x >= 5])
	all_count_ten_negs_found = len([x for x in all_num_neg_found if x >= 10])

	print(f"Overall Statistics:")
	print(f"Total number of negatives: {sum(all_num_neg_found)}")
	print(f"Average number of negatives: {all_avg_num_negs}")
	print(f"Count all positives: {len(all_num_neg_found)}")
	print(f"Count some negs found: {all_count_some_negs_found}")
	print(f"Count >=5 negs found: {all_count_five_negs_found}")
	print(f"Count >=10 negs found: {all_count_ten_negs_found}")
	return


def build_balanced_examples_in_partition(cur_potpos_path, cur_potneg_wordnet_path, cur_potneg_word2vec_path,
										 balanced_samples_negs_per_pos, only_negable_pos_flag, wn_then_w2v_flag,
										 only_wn_entries_flag, neg_source, partition_key, lang, all_preds=None,
										 frequency_bars: list = None, debug=False):
	print(f"building balanced examples in partition: {partition_key}")

	pronouns = ['我', '你', '他', '她', '它', '他们', '她们', '它们', '有人', '自己', '人', 'I', 'you', 'he', 'she', 'it', 'them',
				'someone', 'somebody', 'one', 'myself', 'yourself', 'yourselves', 'himself', 'herself', 'themselves']
	posi_frequency_bucket = {k: 0 for k in sorted(frequency_bars)} if frequency_bars is not None else None

	assert neg_source in ['wordnet', 'wordnet+ext', 'wordnet+w2v', 'word2vec']
	if not only_negable_pos_flag:
		raise NotImplementedError
	if not os.path.exists(cur_potpos_path):
		print(f"current potpos path does not exist for partition {partition_key}, skipping......", file=sys.stderr)
		return {}, posi_frequency_bucket
	if neg_source in ['wordnet', 'wordnet+ext', 'wordnet+w2v'] and not os.path.exists(cur_potneg_wordnet_path):
		print(f"current wordnet potneg path ``{cur_potneg_wordnet_path}'' does not exist for partition {partition_key}, skipping......", file=sys.stderr)
		return {}, posi_frequency_bucket
	if neg_source in ['wordnet+ext', 'wordnet+w2v', 'word2vec'] and not os.path.exists(cur_potneg_word2vec_path):
		print(f"current word2vec potneg path does not exist for partition {partition_key}, skipping......", file=sys.stderr)
		return {}, posi_frequency_bucket

	cur_positives = []
	with open(cur_potpos_path, 'r', encoding='utf8') as pos_fp:
		for line in pos_fp:
			item = json.loads(line)
			cur_positives.append(item)

	cur_negatives = {}

	if neg_source in ['wordnet', 'wordnet+ext', 'wordnet+w2v']:
		with open(cur_potneg_wordnet_path, 'r', encoding='utf8') as negwn_fp:
			for line in negwn_fp:
				item = json.loads(line)
				if item['posi_idx'] not in cur_negatives:
					cur_negatives[item['posi_idx']] = []
				cur_negatives[item['posi_idx']].append(item)
	elif neg_source in ['word2vec']:
		pass
	else:
		raise AssertionError

	if neg_source in ['wordnet+ext', 'wordnet+w2v', 'word2vec']:
		with open(cur_potneg_word2vec_path, 'r', encoding='utf8') as negw2v_fp:
			for line in negw2v_fp:
				item = json.loads(line)
				if item['posi_idx'] not in cur_negatives:
					if only_wn_entries_flag:
						continue
					else:
						cur_negatives[item['posi_idx']] = []
				cur_negatives[item['posi_idx']].append(item)
	elif neg_source in ['wordnet']:
		pass
	else:
		raise AssertionError

	balance_samples = {}  # the positives entries from which some negative entries have been derived.
	num_posi_in_balanced_samples = 0
	num_negi_in_balanced_samples = 0

	print_flag = True

	for posi_idx in cur_negatives:
		cur_dict = {'posi': cur_positives[posi_idx], 'negi': []}
		curpos_upred, curpos_subj, curpos_obj, curpos_tsubj, curpos_tobj = parse_rel(cur_positives[posi_idx])
		if curpos_subj in pronouns or curpos_obj in pronouns:
			continue

		if all_preds is not None and posi_frequency_bucket is not None:
			argtypes = f"{curpos_tsubj}#{curpos_tobj}"  # the type identifiers in all_pred_set are ordered.
			_, posi_predstr = rel2concise_str(curpos_upred, curpos_subj, curpos_obj, curpos_tsubj, curpos_tobj,
											  lang=lang)
			cur_posi_freq = all_preds[argtypes][posi_predstr]['n']
			for thres in posi_frequency_bucket:
				# the thresholds for the buckets are ordered, so by breaking at the first satisfaction of the condition,
				# we are sending the positive entry to the correct bucket. - Teddy
				if cur_posi_freq < thres:
					posi_frequency_bucket[thres] += 1
					break

		cur_available_negs = []
		cur_available_negs_secondary = []

		for neg in cur_negatives[posi_idx]:
			assert neg['posi_upred'] == curpos_upred

			neg_upred, _, _, _, _ = parse_rel(neg)
			if neg_upred == neg['posi_upred']:
				print(f"Unexpected Behavior!")
				continue
			elif print_flag:
				print(f"neg upred: {neg_upred}; pos upred: {neg['posi_upred']};")
				print_flag = False

			assert neg['type'] in ['wn', 'ext', 'non-ext']
			if neg_source == 'wordnet':
				if neg['type'] in ['wn']:
					cur_available_negs.append(neg)
			elif neg_source == 'wordnet+ext':
				if neg['type'] in ['wn']:
					cur_available_negs.append(neg)
				elif neg['type'] in ['ext']:
					if wn_then_w2v_flag:
						cur_available_negs_secondary.append(neg)
					else:
						cur_available_negs.append(neg)
			elif neg_source == 'wordnet+w2v':
				if neg['type'] in ['wn']:
					cur_available_negs.append(neg)
				elif neg['type'] in ['ext', 'non-ext']:
					if wn_then_w2v_flag:
						cur_available_negs_secondary.append(neg)
					else:
						cur_available_negs.append(neg)
			elif neg_source == 'word2vec':
				if neg['type'] in ['ext', 'non-ext']:
					cur_available_negs.append(neg)
			else:
				raise AssertionError

		if len(cur_available_negs) > balanced_samples_negs_per_pos:
			cur_chosen_negs = random.sample(cur_available_negs, balanced_samples_negs_per_pos)
		else:
			cur_chosen_negs = cur_available_negs
			num_remaining_slots = balanced_samples_negs_per_pos - len(cur_chosen_negs)
			if len(cur_available_negs_secondary) > num_remaining_slots:
				cur_chosen_negs += random.sample(cur_available_negs_secondary, num_remaining_slots)
			else:
				cur_chosen_negs += cur_available_negs_secondary
		cur_dict['negi'] = cur_chosen_negs
		balance_samples[posi_idx] = cur_dict
		num_posi_in_balanced_samples += 1
		num_negi_in_balanced_samples += len(cur_chosen_negs)

	assert len(balance_samples) == len(cur_negatives)
	assert posi_frequency_bucket is None or sum(posi_frequency_bucket.values()) == len(balance_samples), \
		print(f"# of samples mismatch: sum of buckets -> {sum(posi_frequency_bucket.values())}; len of samples -> {len(balance_samples)}")
	if debug:
		print(f"posi_frequency_bucket: {posi_frequency_bucket};")
	posi_frequency_bucket = {k: posi_frequency_bucket[k] / float(len(balance_samples)) if len(balance_samples) > 0 \
																					   else posi_frequency_bucket[k] \
							 for k in posi_frequency_bucket} \
		if posi_frequency_bucket is not None else None
	if debug:
		print(f"normalized posi_frequency_bucket: {posi_frequency_bucket};")

	print(f"number of positives in current balanced samples: {num_posi_in_balanced_samples};")
	print(f"number of negatives in current balanced samples: {num_negi_in_balanced_samples}")
	return balance_samples, posi_frequency_bucket


# cleared; untested;
def build_balanced_examples(args, date_slices):
	potexamples_by_partition = {}
	num_posi_by_partition = {}
	for partition_key in date_slices:
		cur_potpos_path = args.potential_positives_path % partition_key
		cur_potneg_wordnet_path = args.potential_negatives_path % ('wordnet', partition_key)
		cur_potneg_word2vec_path = args.potential_negatives_path % ('word2vec', partition_key)

		cur_balance_samples, _ = build_balanced_examples_in_partition(cur_potpos_path, cur_potneg_wordnet_path,
																   cur_potneg_word2vec_path,
																   args.balanced_samples_negs_per_pos,
																   args.only_negable_pos_flag,
																   args.wn_then_w2v_flag, args.wn_only_entries_flag,
																   args.neg_source, partition_key, args.lang)
		potexamples_by_partition[partition_key] = cur_balance_samples
		# the size of ``cur_balance_samples'' is equal to the number of positives that are taken into consideration, since this is a dictionary where its keys are posi_idxes
		num_posi_by_partition[partition_key] = len(cur_balance_samples)

	total_num_posis = 0.0
	for key in num_posi_by_partition:
		total_num_posis += num_posi_by_partition[key]
	if total_num_posis < args.pos_size_of_final_sample:
		notenough_samples = True
	else:
		notenough_samples = False

	print(f"Total number of positive samples in any partition, found with some negatives: {total_num_posis}.")

	num_posisamples_by_partition = {}
	total_num_actually_chosen_samples = 0
	for key in num_posi_by_partition:
		cur_num_posisamples = math.ceil(args.pos_size_of_final_sample * (num_posi_by_partition[key] / total_num_posis))
		if notenough_samples:
			cur_num_posisamples = num_posi_by_partition[key]
		else:
			assert cur_num_posisamples <= num_posi_by_partition[key]
		num_posisamples_by_partition[key] = cur_num_posisamples
		total_num_actually_chosen_samples += cur_num_posisamples

	chosen_samples_dev = []
	chosen_samples_test = []
	assert len(potexamples_by_partition) == len(
		num_posisamples_by_partition)  # assert that the number of partitions are the same
	partition_pool = list(potexamples_by_partition.keys())
	random.shuffle(partition_pool)

	accumulated_numsamples_dev = 0  # a counter, when this is about to exceed half the ``total_num_actually_chosen_samples'', we switch from dev to test
	for key in partition_pool:
		if (accumulated_numsamples_dev + num_posisamples_by_partition[key]) < total_num_actually_chosen_samples / 2.0:
			dump_to_dev_flag = True
			accumulated_numsamples_dev += num_posisamples_by_partition[key]
		else:
			dump_to_dev_flag = False
		cur_final_sample_posi_idxes = random.sample(list(potexamples_by_partition[key].keys()),
													k=num_posisamples_by_partition[key])
		for posi_idx in cur_final_sample_posi_idxes:
			posi_ent = potexamples_by_partition[key][posi_idx]['posi']
			negi_ents = potexamples_by_partition[key][posi_idx]['negi']
			posi_ent['label'] = True
			if dump_to_dev_flag:
				chosen_samples_dev.append(posi_ent)
			else:
				chosen_samples_test.append(posi_ent)
			for negi_ent in negi_ents:
				negi_ent['label'] = False
				if dump_to_dev_flag:
					chosen_samples_dev.append(negi_ent)
				else:
					chosen_samples_test.append(negi_ent)

	print(f"Dumping dev set final samples to {args.final_samples_dev_fn}...")
	with open(args.final_samples_dev_fn, 'w', encoding='utf8') as dev_fp:
		for dev_sample in chosen_samples_dev:
			out_line = json.dumps(dev_sample, ensure_ascii=False)
			dev_fp.write(out_line + '\n')

	print(f"Dumping test set final samples to {args.final_samples_test_fn}...")
	with open(args.final_samples_test_fn, 'w', encoding='utf8') as test_fp:
		for test_sample in chosen_samples_test:
			out_line = json.dumps(test_sample, ensure_ascii=False)
			test_fp.write(out_line + '\n')
	print(f"Finished.")


# cleared; untested;
def build_balanced_examples_with_freqmap(args, date_slices, all_preds):
	frequency_bars = [30, 60, 100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 15000,
					  20000, 30000, 50000, 100000, 100000000]

	potexamples_by_partition = {}
	num_posi_by_partition = {}
	posi_freq_buckets = {}
	for partition_key in date_slices:
		cur_potpos_path = args.potential_positives_path % partition_key
		cur_potneg_wordnet_path = args.potential_negatives_path % ('wordnet', partition_key)
		cur_potneg_word2vec_path = args.potential_negatives_path % ('word2vec', partition_key)

		cur_balance_samples, cur_posi_freq_bucket = build_balanced_examples_in_partition(cur_potpos_path,
																   cur_potneg_wordnet_path, cur_potneg_word2vec_path,
																   args.balanced_samples_negs_per_pos,
																   args.only_negable_pos_flag,
																   args.wn_then_w2v_flag, args.wn_only_entries_flag,
																   args.neg_source, partition_key, args.lang, all_preds,
																   frequency_bars, debug=args.debug)
		potexamples_by_partition[partition_key] = cur_balance_samples
		# the size of ``cur_balance_samples'' is equal to the number of positives that are taken into consideration, since this is a dictionary where its keys are posi_idxes
		num_posi_by_partition[partition_key] = len(cur_balance_samples)
		posi_freq_buckets[partition_key] = cur_posi_freq_bucket

	total_num_posis = 0.0
	for key in num_posi_by_partition:
		total_num_posis += num_posi_by_partition[key]
	if total_num_posis < args.pos_size_of_final_sample:
		notenough_samples = True
	else:
		notenough_samples = False

	print(f"Total number of positive samples in any partition, found with some negatives: {total_num_posis}.")

	num_posisamples_by_partition = {}
	total_num_actually_chosen_samples = 0
	for key in num_posi_by_partition:
		cur_num_posisamples = math.ceil(args.pos_size_of_final_sample * (num_posi_by_partition[key] / total_num_posis))
		if notenough_samples:
			cur_num_posisamples = num_posi_by_partition[key]
		else:
			assert cur_num_posisamples <= num_posi_by_partition[key]
		num_posisamples_by_partition[key] = cur_num_posisamples
		total_num_actually_chosen_samples += cur_num_posisamples

	chosen_samples_dev = []
	chosen_samples_test = []
	assert len(potexamples_by_partition) == len(
		num_posisamples_by_partition)  # assert that the number of partitions are the same
	partition_pool = list(potexamples_by_partition.keys())
	random.shuffle(partition_pool)

	accumulated_numsamples_dev = 0  # a counter, when this is about to exceed half the ``total_num_actually_chosen_samples'', we switch from dev to test
	for key in partition_pool:
		if (accumulated_numsamples_dev + num_posisamples_by_partition[key]) < total_num_actually_chosen_samples / 2.0:
			dump_to_dev_flag = True
			accumulated_numsamples_dev += num_posisamples_by_partition[key]
		else:
			dump_to_dev_flag = False

		cur_negfreq_bucket_portfolio = {k: v * num_posisamples_by_partition[key] * 1.9 for (k, v) in posi_freq_buckets[key].items()}
		# round the portfolio up or down by chance:
		for thres in cur_negfreq_bucket_portfolio:
			residue = cur_negfreq_bucket_portfolio[thres] - math.floor(cur_negfreq_bucket_portfolio[thres])
			this_rho = random.random()
			if this_rho < residue:
				cur_negfreq_bucket_portfolio[thres] = math.ceil(cur_negfreq_bucket_portfolio[thres])
			else:
				cur_negfreq_bucket_portfolio[thres] = math.floor(cur_negfreq_bucket_portfolio[thres])

		if args.debug:
			print(f"key: {key}; negfreq portfolio: {cur_negfreq_bucket_portfolio};")
		posi_idx_iterator = list(potexamples_by_partition[key].keys())
		random.shuffle(posi_idx_iterator)
		cur_final_sample_posi_idxes = []
		for this_posi_idx in posi_idx_iterator:
			negi_bkts = {k: 0 for k in cur_negfreq_bucket_portfolio}
			bad_sample_flag = False
			this_posi_upred, this_posi_subj, this_posi_obj, this_posi_tsubj, this_posi_tobj = parse_rel(potexamples_by_partition[key][this_posi_idx]['posi'])
			this_posi_argtypes = f"{this_posi_tsubj}#{this_posi_tobj}"
			_, this_posi_predstr = rel2concise_str(this_posi_upred, this_posi_subj, this_posi_obj, this_posi_tsubj,
													   this_posi_tobj, lang=args.lang)
			this_posi_freq = all_preds[this_posi_argtypes][this_posi_predstr]['n']

			for this_negid, this_negi in enumerate(potexamples_by_partition[key][this_posi_idx]['negi']):
				this_negi_upred, this_negi_subj, this_negi_obj, this_negi_tsubj, this_negi_tobj = parse_rel(this_negi)
				if this_negi_upred == this_posi_upred:
					raise AssertionError
				this_negi_argtypes = f"{this_negi_tsubj}#{this_negi_tobj}"
				_, this_negi_predstr = rel2concise_str(this_negi_upred, this_negi_subj, this_negi_obj, this_negi_tsubj,
													   this_negi_tobj, lang=args.lang)
				this_negi_freq = all_preds[this_negi_argtypes][this_negi_predstr]['n']
				if 0 < args.freq_multiples_cap < this_posi_freq / this_negi_freq:
					bad_sample_flag = True
				elif 0 < args.freq_multiples_cap < this_negi_freq / this_posi_freq:
					bad_sample_flag = True
				else:
					pass
				for thres in cur_negfreq_bucket_portfolio:
					if this_negi_freq < thres:
						negi_bkts[thres] += 1
						break

			for thres in cur_negfreq_bucket_portfolio:
				if cur_negfreq_bucket_portfolio[thres] - negi_bkts[thres] < 0:
					bad_sample_flag = True
					break

			if not bad_sample_flag:
				cur_final_sample_posi_idxes.append(this_posi_idx)
				for thres in cur_negfreq_bucket_portfolio:
					cur_negfreq_bucket_portfolio[thres] -= negi_bkts[thres]

		for posi_idx in cur_final_sample_posi_idxes:
			posi_ent = potexamples_by_partition[key][posi_idx]['posi']
			negi_ents = potexamples_by_partition[key][posi_idx]['negi']
			posi_ent['label'] = True
			if dump_to_dev_flag:
				chosen_samples_dev.append(posi_ent)
			else:
				chosen_samples_test.append(posi_ent)
			for negi_ent in negi_ents:
				negi_ent['label'] = False
				if dump_to_dev_flag:
					chosen_samples_dev.append(negi_ent)
				else:
					chosen_samples_test.append(negi_ent)

	print(f"Dumping dev set final samples to {args.final_samples_dev_fn}...")
	with open(args.final_samples_dev_fn, 'w', encoding='utf8') as dev_fp:
		for dev_sample in chosen_samples_dev:
			out_line = json.dumps(dev_sample, ensure_ascii=False)
			dev_fp.write(out_line + '\n')

	print(f"Dumping test set final samples to {args.final_samples_test_fn}...")
	with open(args.final_samples_test_fn, 'w', encoding='utf8') as test_fp:
		for test_sample in chosen_samples_test:
			out_line = json.dumps(test_sample, ensure_ascii=False)
			test_fp.write(out_line + '\n')
	print(f"Finished.")


# cleared, not tested
def postprocess_balanced_examples(args):
	entries = []

	print(f"post processing dev set...")
	with open(args.final_samples_dev_fn, 'r', encoding='utf8') as dev_fp:
		for lidx, line in enumerate(dev_fp):
			if lidx % 10000 == 0:
				print(lidx)
			item = json.loads(line)
			rel_sent = reconstruct_sent_from_rel(item, 1000, lang=args.lang)
			item['proposition'] = rel_sent
			entries.append(item)
	with open(args.final_samples_dev_fn, 'w', encoding='utf8') as new_dev_fp:
		for item in entries:
			out_line = json.dumps(item, ensure_ascii=False)
			new_dev_fp.write(out_line + '\n')

	del entries
	entries = []

	print(f"post processing test set...")
	with open(args.final_samples_test_fn, 'r', encoding='utf8') as test_fp:
		for lidx, line in enumerate(test_fp):
			if lidx % 10000 == 0:
				print(lidx)
			item = json.loads(line)
			rel_sent = reconstruct_sent_from_rel(item, 1000, lang=args.lang)
			item['proposition'] = rel_sent
			entries.append(item)

	with open(args.final_samples_test_fn, 'w', encoding='utf8') as new_test_fp:
		for item in entries:
			out_line = json.dumps(item, ensure_ascii=False)
			new_test_fp.write(out_line + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--positives_base', default='clue_positives_%s_%s.json', type=str,
						help='two variables: version name; [dev/test]')
	parser.add_argument('--negatives_base', default='clue_negatives_%s_%d_%d_%s_%s.json', type=str,
						help='three variables: version name; [wordnet/word2vec]; [dev/test]')
	parser.add_argument('--potential_positives_path', default='../clue_time_slices/clue_potential_positives_%s_%s.json',
						type=str)
	parser.add_argument('--potential_negatives_path',
						default='../clue_time_slices/clue_potential_negatives_%s_%d_%d_%s_%s.json', type=str)
	parser.add_argument('--time_interval', default=3, type=int)
	parser.add_argument('--version', default='5_10_triple_doc_100_disjoint', type=str)
	parser.add_argument('--do_load_triples', action='store_true')
	parser.add_argument('--do_compute_upred_vecs', action='store_true')
	parser.add_argument('--do_compute_posi_synsets', action='store_true')
	parser.add_argument('--do_wordnet', action='store_true',
						help='Use wordnet hyponyms and troponyms to generate negatives.')
	parser.add_argument('--do_word2vec', action='store_true',
						help='Use word2vec BagOfWord Similarities to generate negatives.')
	parser.add_argument('--which_positives', default='sampled',
						help='find negatives for [sampled] or [potential] or [single_potential] (just do generate_negative '
							 'for a single partition) positives.')
	parser.add_argument('--single_pot_partition_idx', default=None, type=int,
						help='the partition index of the partition for which to generate negatives, if we are to generate '
							 'negatives for a single partition.')
	parser.add_argument('--global_presence', default='pred', type=str,
						help='how global presence is counted: by presence of triple or by presence of bare predicates: [triple/pred]')
	parser.add_argument('--vnonly', action='store_true',
						help='whether to use WordNet entries with `v` and `n` POS-tags only, useful only for the WordNet setting.')
	parser.add_argument('--firstonly', action='store_true',
						help='whether to use only the first synsets of an WordNet entry, useful only for the WordNet setting.')
	parser.add_argument('--wsd_model_dir', default='', type=str)
	parser.add_argument('--wordnet_dir', default='../generic_aux_data/wn-cmn-lmf.xml', type=str)
	parser.add_argument('--word2vec_path', default='../generic_aux_data/sgns.merge.char', type=str)
	parser.add_argument('--wsd_batch_size', default=32, type=int)
	parser.add_argument('--all_triples_path', default='../../clue_typed_triples_tacl.json', type=str)
	parser.add_argument('--partition_triples_path', default='../clue_time_slices/clue_typed_triples_%s_%s.json',
						type=str)
	parser.add_argument('--triple_set_path', default='../clue_inter_data/clue_all_triple_set.json', type=str)
	parser.add_argument('--pred_set_path', default='../clue_inter_data/clue_all_pred_set.json', type=str)
	parser.add_argument('--pred_vectors_path', default='../clue_inter_data/clue_triple_vectors.h5', type=str)
	parser.add_argument('--pred_vectors_cache_size', default=300000, type=int)
	parser.add_argument('--max_num_lines', default=-1, type=int)
	parser.add_argument('--global_presence_thres', default=5, type=int)
	parser.add_argument('--global_presence_cap', default=0, type=int, help='maximum number of occurrences for negatives,'
																		   'used to avoid over-general words; if set to'
																		   '0, means no cap.')

	parser.add_argument('--similarity_thres', default=0.75, type=float,
						help='hard threshold for word2vec similarity, useful only for Word2Vec setting.')
	parser.add_argument('--similarity_cap', default=0.9, type=float,
						help='hard cap for word2vec similarity, too similar predicates are considered false negatives and thrown out, useful only for Word2Vec setting.')
	parser.add_argument('--ranking_thres', default=7, type=int,
						help='top-{thres} similar negatives are within the range of consideration.')
	parser.add_argument('--ranking_cap', default=1, type=int,
						help='Except for the top-{cap} similar negatives, which we fear are in fact false positives.')
	parser.add_argument('--negs_per_pos', default=3, type=int,
						help='number of negatives per positive, useful only for Word2Vec setting.')
	parser.add_argument('--population_size', default=5000, type=int,
						help='size of predicate pool from which to find and sample challenging negatives, useful only for Word2Vec setting.')
	parser.add_argument('--max_num_posi_collected_per_partition', default=30000, type=int,
						help='Maximum number of positives-with-negatives from which to select the negatives.')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--ext_ratio', default=0.5, type=int)

	parser.add_argument('--do_final_sampling', action='store_true', help='')
	parser.add_argument('--do_final_sampling_freq_map', action='store_true', help='')
	parser.add_argument('--do_postprocessing', action='store_true', help='')
	parser.add_argument('--balanced_samples_negs_per_pos', default=1, type=int,
						help='Number of negatives per positive in the final balanced samples.')
	parser.add_argument('--only_negable_pos_flag', action='store_true',
						help='whether or not to only select those positives that have some negatives paired with them.')
	parser.add_argument('--wn_then_w2v_flag', action='store_true', help='whether or not to use lexicographic order: '
																		'first wordnet then word2vec(-ext) when selecting '
																		'negatives to final sample.')

	parser.add_argument('--allow_prep_mismatch_replacements', action='store_true',
						help='whether or not to allow for preposition mismatches in the'
							 'wordnet hyponym replacements as potential negatives')
	parser.add_argument('--allow_backoff_wsd', action='store_true',
						help='whether or not to backoff to selecting from all synsets of a token when the wsd system returns null.')

	parser.add_argument('--wn_only_entries_flag', action='store_true', help='')
	parser.add_argument('--neg_source', default='wordnet', type=str,
						help='[wordnet / wordnet+ext / wordnet+w2v / word2vec]')
	parser.add_argument('--pos_size_of_final_sample', default=1400000, type=int,
						help='This is only the positives, not counting negatives; but this is also the number of entries for dev & test sets combined.')
	parser.add_argument('--final_samples_base_fn', default='clue_final_samples_%s_%d_%d_%d_%d_%s_%s_%.1f_%s.json', type=str,
						help='six variables: version_without_samplesize; final_sample_size; balanced_samples_negs_per_pos; [joint/lexic]; neg_source; [dev/test].')
	parser.add_argument('--lang', required=True, type=str, help='[zh / en]')
	parser.add_argument('--no_cuda', action='store_true', help="Whether or not to force not using cuda backend, effective only for computing synsets.")

	# parser.add_argument('--ref_freqmap_path', type=str, default='', help='Path to reference frequency map (from free positives), useful only for "do_final_sampling_freq_map"')
	parser.add_argument('--freq_multiples_cap', type=float, default=0, help='maximum multiples of frequency between positives and their corresponding negatives (two-ways); '
																			'when set to 0, means no cap.')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--global_triple_absence_flag', action='store_true', help='whether or not to use global triple absence as a negative filter.')

	random.seed()
	args = parser.parse_args()
	disjoint_str = args.version.split('_')[-1]
	version_name_for_potpos = args.version

	assert args.global_presence in ['pred', 'triple']
	assert not args.global_triple_absence_flag or args.global_presence != "triple", "global_triple_absence_flag is only applicable when global_presence is not triple."
	assert disjoint_str in ['disjoint', 'sliding']
	if disjoint_str == 'disjoint':
		args.disjoint_window = True
	elif disjoint_str == 'sliding':
		args.disjoint_window = False
	else:
		raise AssertionError

	if args.wn_then_w2v_flag is True:
		args.ordering_of_combination = 'lexic'
	else:
		args.ordering_of_combination = 'joint'

	assert args.global_presence_cap == 0 or args.global_presence_cap > args.global_presence_thres

	args.partition_triples_path = args.partition_triples_path % (disjoint_str, '%s')
	args.positives_dev_fn = args.positives_base % (args.version, 'dev')
	args.positives_test_fn = args.positives_base % (args.version, 'test')
	args.negatives_dev_fn = args.negatives_base % (args.version, args.global_presence_thres, args.global_presence_cap, '%s', 'dev')
	args.negatives_test_fn = args.negatives_base % (args.version, args.global_presence_thres, args.global_presence_cap, '%s', 'test')

	args.final_samples_dev_fn = args.final_samples_base_fn % (version_name_for_potpos, args.global_presence_thres,
															  args.global_presence_cap, args.pos_size_of_final_sample,
															  args.balanced_samples_negs_per_pos,
															  args.ordering_of_combination,
															  args.neg_source, args.freq_multiples_cap, 'dev')
	args.final_samples_test_fn = args.final_samples_base_fn % (version_name_for_potpos, args.global_presence_thres,
															   args.global_presence_cap, args.pos_size_of_final_sample,
															   args.balanced_samples_negs_per_pos,
															   args.ordering_of_combination,
															   args.neg_source, args.freq_multiples_cap, 'test')

	args.potential_positives_path = args.potential_positives_path % (version_name_for_potpos, '%s')
	args.potential_negatives_path = args.potential_negatives_path % (version_name_for_potpos, args.global_presence_thres,
																	 args.global_presence_cap, '%s', '%s')

	datemngr = DateManager(lang=args.lang)
	if args.disjoint_window:
		date_slices, _ = datemngr.setup_dateslices(args.time_interval)
	else:
		date_slices, _ = datemngr.setup_dates(args.time_interval)

	if args.which_positives == 'single_potential' and args.single_pot_partition_idx is not None:
		if args.single_pot_partition_idx >= len(date_slices):
			print(
				f"Process single potential with time slice index {args.single_pot_partition_idx} out of range for date_slices of size {len(date_slices)}, exitting......")
			exit(0)
		this_potpos_path = args.potential_positives_path % date_slices[args.single_pot_partition_idx]
		if os.path.getsize(this_potpos_path) < 100:
			print(
				f"Process single potential with time slice: ``{date_slices[args.single_pot_partition_idx]}'' with size < 100 bytes, exitting......")
			exit(0)

	if args.do_final_sampling:
		build_balanced_examples(args, date_slices)
		postprocess_balanced_examples(args)
		exit(0)
	if args.do_postprocessing:
		postprocess_balanced_examples(args)
		exit(0)

	if args.do_compute_posi_synsets:
		assert args.lang in ['en']
		if args.which_positives == 'single_potential' and args.single_pot_partition_idx is not None:
			print(f"Computing posi synsets only for slice index: {args.single_pot_partition_idx}!")
			dslice_idx = args.single_pot_partition_idx
		else:
			print(f"Computing posi synsets for all slice indices!")
			dslice_idx = None
		compute_posi_synsets(args.potential_positives_path, wsd_model_dir=args.wsd_model_dir, date_slices=date_slices,
							 batch_size=args.wsd_batch_size, lang=args.lang, date_slice_idx=dslice_idx, no_cuda=args.no_cuda)
		exit(0)

	if args.do_load_triples:
		if args.global_presence == 'pred':
			print(f"Loading only predicate set! If you need to do global_presence by triple, please set it so and "
				  f"re-run this do_load_triples step!")
			if args.global_triple_absence_flag is False:
				_, all_preds = load_triple_set(args.all_triples_path, args.triple_set_path, args.pred_set_path,
											   args.max_num_lines, verbose=args.verbose, lang=args.lang, only_preds=True)
				global_tplstrs = None
			else:
				global_tplstrs, all_preds = load_triple_set(args.all_triples_path, args.triple_set_path, args.pred_set_path,
															args.max_num_lines, verbose=args.verbose, lang=args.lang,
															only_preds=False, only_tplstrs=True)
			sound_tplstrs = None
		elif args.global_presence == 'triple':
			all_triples, all_preds = load_triple_set(args.all_triples_path, args.triple_set_path, args.pred_set_path,
													 args.max_num_lines, verbose=args.verbose, lang=args.lang)
			sound_tplstrs = filter_sound_triples(all_triples, args.global_presence_thres, args.global_presence_cap)
			global_tplstrs = None
			del all_triples
		else:
			raise AssertionError
		lemm_2_predstr = None
		lemm_noprep_2_predstr = None
	else:
		# if we are certain this is not going to be used, then don't load it: the thing is pretty big!
		if args.global_presence == 'pred' and (
				args.do_wordnet or args.do_word2vec or args.do_final_sampling or args.do_final_sampling_freq_map or args.do_compute_upred_vecs):
			sound_tplstrs = set()
			if args.global_triple_absence_flag is True:
				global_tplstrs = set()
				with open(args.triple_set_path, 'r', encoding='utf8') as fp:
					for lidx, line in enumerate(fp):
						if lidx % 100000 == 0:
							print(f"Loading global triple set: {lidx} lines loaded!")
						assert line[-1] == '\n'
						line = line[:-1]
						global_tplstrs.add(line)
			else:
				global_tplstrs = None
		else:
			all_triples = dict()
			print(f"Loading triple strings from: {args.triple_set_path}")
			with open(args.triple_set_path, 'r', encoding="utf8") as fp:
				for line in fp:
					item = json.loads(line)
					assert item['tplstr'] not in all_triples
					all_triples[item['tplstr']] = {'r': item['r'], 'n': item['n']}

			sound_tplstrs = filter_sound_triples(all_triples, args.global_presence_thres, args.global_presence_cap)
			global_tplstrs = None
			del all_triples

		print(f"Loading predicate strings from: {args.pred_set_path}")
		all_preds = dict()
		lemm_2_predstr = dict()
		lemm_noprep_2_predstr = dict()
		count_tpreds_over_thres = 0
		with open(args.pred_set_path, 'r', encoding='utf8') as fp:
			for lidx, line in enumerate(fp):
				if lidx % 100000 == 0:
					print(f"Loading predicate set: {lidx} lines loaded!")
				item = json.loads(line)
				assert item['type'] is not None
				if item['type'] not in all_preds:
					all_preds[item['type']] = {}
				if item['type'] not in lemm_2_predstr:
					lemm_2_predstr[item['type']] = {}
				if item['type'] not in lemm_noprep_2_predstr:
					lemm_noprep_2_predstr[item['type']] = {}

				if item['lemm_predstr'] not in lemm_2_predstr[item['type']]:
					lemm_2_predstr[item['type']][item['lemm_predstr']] = []
				if item['lemm_noprep_predstr'] not in lemm_noprep_2_predstr[item['type']]:
					lemm_noprep_2_predstr[item['type']][item['lemm_noprep_predstr']] = []

				# these lemm_2_predstr and lemm_noprep_2_predstr information is stored regardless of whether the predstr
				# satisfies the num_occ thresholds.
				assert item['predstr'] not in lemm_2_predstr[item['type']][item['lemm_predstr']]
				lemm_2_predstr[item['type']][item['lemm_predstr']].append(item['predstr'])
				assert item['predstr'] not in lemm_noprep_2_predstr[item['type']][item['lemm_noprep_predstr']]
				lemm_noprep_2_predstr[item['type']][item['lemm_noprep_predstr']].append(item['predstr'])

				assert item['predstr'] not in all_preds[item['type']]
				if fits_thresholds(item['n'], args.global_presence_thres, args.global_presence_cap):
					count_tpreds_over_thres += 1
				else:
					continue  # TODO: only include a predicate into ``all_preds'' if its occurrence satisfies thresholds! Saves time.
				if 'global_id' in item:
					all_preds[item['type']][item['predstr']] = {'p': item['p'], 'n': item['n'],
																'global_id': item['global_id'],
																'predstr': item['predstr'],
																'lemm_predstr': item['lemm_predstr'],
																'lemm_noprep_predstr': item['lemm_noprep_predstr']}
				else:
					# 'global_id' not being in 'item' means this loading is for doing compute_upred_vecs
					all_preds[item['type']][item['predstr']] = {'p': item['p'], 'n': item['n'],
																'predstr': item['predstr'],
																'lemm_predstr': item['lemm_predstr'],
																'lemm_noprep_predstr': item['lemm_noprep_predstr']}
		print(f"Count typed predicates between thresholds {args.global_presence_thres} and {args.global_presence_cap}: {count_tpreds_over_thres}")

	if args.do_final_sampling_freq_map:
		build_balanced_examples_with_freqmap(args, date_slices, all_preds)
		postprocess_balanced_examples(args)
		exit(0)

	if args.do_compute_upred_vecs:
		word_vectors = readWord2Vec(args.word2vec_path, verbose=args.verbose, lang=args.lang)
		h5file = h5py.File(args.pred_vectors_path, 'w')
		pred_vectors, mismatches_dset = build_upred_vectors_h5py(all_preds, word_vectors, h5file, args.pred_set_path,
																 lang=args.lang)
		h5file.close()
		exit(0)

	if args.which_positives == 'single_potential':
		assert args.single_pot_partition_idx is not None

	if args.do_wordnet:
		if args.lang == 'zh':
			wordnet_dict = readWordNet(args.wordnet_dir)
		elif args.lang == 'en':
			wordnet_dict = None
		else:
			raise AssertionError
		if args.global_presence != 'triple':
			del sound_tplstrs
			sound_tplstrs = None
		if args.which_positives == 'sampled':
			find_negatives_wrapper(args, wordnet_dict, None, None, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
								   lemm_noprep_2_predstr, do_which='wordnet')
		elif args.which_positives == 'potential':
			find_potential_negatives(args, wordnet_dict, None, None, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
									 lemm_noprep_2_predstr, date_slices, do_which='wordnet')
		elif args.which_positives == 'single_potential':
			find_potential_negatives(args, wordnet_dict, None, None, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
									 lemm_noprep_2_predstr, date_slices, do_which='wordnet',
									 single_pot_partition_idx=args.single_pot_partition_idx)
		else:
			raise AssertionError
	if args.do_word2vec:
		word_vectors = readWord2Vec(args.word2vec_path, verbose=False, lang=args.lang)
		h5file = h5py.File(args.pred_vectors_path, 'r')
		if args.global_presence != 'triple':
			del sound_tplstrs
			sound_tplstrs = None
		if args.which_positives == 'sampled':
			find_negatives_wrapper(args, None, h5file, word_vectors, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
								   lemm_noprep_2_predstr, do_which='word2vec')
		elif args.which_positives == 'potential':
			find_potential_negatives(args, None, h5file, word_vectors, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
									 lemm_noprep_2_predstr, date_slices, do_which='word2vec')
		elif args.which_positives == 'single_potential':
			find_potential_negatives(args, None, h5file, word_vectors, global_tplstrs, sound_tplstrs, all_preds, lemm_2_predstr,
									 lemm_noprep_2_predstr, date_slices, do_which='word2vec',
									 single_pot_partition_idx=args.single_pot_partition_idx)
		else:
			raise AssertionError
