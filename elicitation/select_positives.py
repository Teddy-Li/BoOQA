import json
import argparse
import os
from math import floor, ceil
import random
import copy
from qaeval_utils import DateManager, parse_rel, parse_time, upred2bow, check_vague


def filter_fullcorpus_predicates(sents, pred_thres, pred_cap, filter_mode, lang):
	assert pred_cap == 0 or pred_cap - pred_thres >= 2
	print(f"Filtering predicates from the whole corpus with threshold: {pred_thres}, cap: {pred_cap}, filtering by {filter_mode}!")
	pred_dict = {}

	for sidx, item in enumerate(sents):
		if 0 <= args.max_num_lines < sidx:
			break
		if sidx % 100000 == 0:
			print(f"sidx: {sidx}")
		this_sent_preds = set()
		try:
			item = json.loads(item)
		except json.decoder.JSONDecodeError as e:
			print(f"Skiped line: ``{item}''")
			continue
		aid = item['articleId']

		for rel in item["rels"]:
			upred, subj, obj, tsubj, tobj = parse_rel(rel)
			tpred = f"{upred}::{tsubj}::{tobj}"
			if filter_mode == 'triple':
				if tpred not in pred_dict:
					pred_dict[tpred] = 0
				pred_dict[tpred] += 1
			elif filter_mode in ['sent', 'doc']:
				this_sent_preds.add(tpred)
			else:
				raise AssertionError

		if filter_mode == 'triple':
			pass
		elif filter_mode == 'sent':
			for tpred in this_sent_preds:
				if tpred not in pred_dict:
					pred_dict[tpred] = 0
				pred_dict[tpred] += 1
		elif filter_mode == 'doc':
			for tpred in this_sent_preds:
				if tpred not in pred_dict:
					pred_dict[tpred] = set()
				pred_dict[tpred].add(aid)
		else:
			raise AssertionError

	toy_thresholds = (3, 30)
	for toy_thres in range(toy_thresholds[0], toy_thresholds[1]):
		accepted_preds = []
		for tpred in pred_dict:
			if filter_mode in ['triple', 'sent']:
				if pred_dict[tpred] > toy_thres and (pred_cap == 0 or pred_dict[tpred] < pred_cap):
					accepted_preds.append(tpred)
			elif filter_mode in ['doc']:
				if len(pred_dict[tpred]) > toy_thres and (pred_cap == 0 or len(pred_dict[tpred]) < pred_cap):
					accepted_preds.append(tpred)
			else:
				raise AssertionError
		print(f"# of accepted predicates for threshold {toy_thres}, cap {pred_cap}: {len(accepted_preds)}!")

	accepted_preds = []
	for tpred in pred_dict:
		if filter_mode in ['triple', 'sent']:
			if pred_dict[tpred] > pred_thres and (pred_cap == 0 or pred_dict[tpred] < pred_cap):
				accepted_preds.append(tpred)
		elif filter_mode in ['doc']:
			if len(pred_dict[tpred]) > pred_thres and (pred_cap == 0 or len(pred_dict[tpred]) < pred_cap):
				accepted_preds.append(tpred)
		else:
			raise AssertionError
	print(f"# of accepted predicates for threshold {pred_thres}, cap {pred_cap}: {len(accepted_preds)}!")

	return accepted_preds


def filter_partition_entpairs(cur_bucket, ep_thres, ep_cap, filter_mode, partition_key, lang):
	print(f"Filtering entity pairs for partition: {partition_key}, with threshold: {ep_thres}, cap: {ep_cap}, filtering by: {filter_mode}!")
	entpair_dict = {}
	entpair_preds = {}
	pronouns = ['我', '你', '他', '她', '它', '他们', '她们', '它们', '有人', '自己', '人', 'i', 'you', 'he', 'she', 'it', 'them',
				'someone', 'somebody', 'one', 'myself', 'yourself', 'yourselves', 'himself', 'herself', 'themselves', 'people',
				'man', 'woman', 'her', 'him', 'her', 'his', 'hers', 'his', 'our', 'ours', 'your', 'yours', 'their', 'theirs',
				'its', 'anyone', 'anybody', 'every', 'everyone', 'everybody']
	mightbe_pronouns = ['people']

	for sidx, item in enumerate(cur_bucket):  # each item is one sentence parsed.
		# if sidx % 10000 == 0:
		# 	print(f"{sidx}")
		this_sent_entpairs = set()
		aid = item['articleId']

		for rel in item["rels"]:
			upred, subj, obj, tsubj, tobj = parse_rel(rel)
			subj = subj.lower()
			obj = obj.lower()
			upred = upred.lower()
			if subj.lower() in pronouns or obj.lower() in pronouns:
				continue
			elif subj.lower() in mightbe_pronouns or obj.lower() in mightbe_pronouns:
				print(f"Relation with might-be pronouns: {rel}")
			entpair = f"{subj}::{obj}::{tsubj}::{tobj}"
			if filter_mode == 'triple':
				if entpair not in entpair_dict:
					entpair_dict[entpair] = 0
				entpair_dict[entpair] += 1
			elif filter_mode in ['sent', 'doc']:
				this_sent_entpairs.add(entpair)
			else:
				raise AssertionError
			if entpair not in entpair_preds:
				entpair_preds[entpair] = set()
			entpair_preds[entpair].add(upred)

		if filter_mode == 'triple':
			pass
		elif filter_mode == 'sent':
			for entpair in this_sent_entpairs:
				if entpair not in entpair_dict:
					entpair_dict[entpair] = 0
				entpair_dict[entpair] += 1
		elif filter_mode == 'doc':
			for entpair in this_sent_entpairs:
				if entpair not in entpair_dict:
					entpair_dict[entpair] = set()
				entpair_dict[entpair].add(aid)
		else:
			raise AssertionError

	accepted_eps = []
	for ep in entpair_dict:
		assert ep in entpair_preds
		# for whichever filter_mode, the entity pair must have been seen with more than ep_thres number of unique predicates,
		# so as to have a diverse pool of context.
		if not (len(entpair_preds[ep]) >= ep_thres and (ep_cap is None or ep_cap == 0 or len(entpair_preds[ep]) < ep_cap)):
			continue

		if filter_mode in ['triple', 'sent']:
			if entpair_dict[ep] >= ep_thres and (ep_cap is None or ep_cap == 0 or entpair_dict[ep] < ep_cap):
				accepted_eps.append(ep)
		elif filter_mode in ['doc']:
			if len(entpair_dict[ep]) >= ep_thres and (ep_cap is None or ep_cap == 0 or len(entpair_dict[ep]) < ep_cap):
				accepted_eps.append(ep)
		else:
			raise AssertionError

	print(f"# of accepted entity pairs for partition {partition_key} at threshold {ep_thres}: {len(accepted_eps)}; cap: {ep_cap}")
	return accepted_eps


def filter_positive_triples(cur_bucket, accepted_preds, accepted_eps, partition_key, lang):

	potential_positives = []
	sents_with_potpos = set()
	all_num_rels = 0

	for sidx, item in enumerate(cur_bucket):
		if sidx % 10000 == 0 and sidx > 0:
			print(f"Partition key {partition_key} sidx: {sidx}; number of potential positives: {len(potential_positives)}")

		all_num_rels += len(item['rels'])
		for rel in item["rels"]:
			upred, subj, obj, tsubj, tobj = parse_rel(rel)
			tpred = f"{upred}::{tsubj}::{tobj}"
			entpair = f"{subj}::{obj}::{tsubj}::{tobj}"

			# skip the too vague predicates such as ``是'' or ``有''
			upred_bow = upred2bow(upred, lang=lang)
			if check_vague(upred_bow) is True:
				continue

			if tpred in accepted_preds and entpair in accepted_eps:
				reldct = {'r': rel["r"], 'partition_key': partition_key, 'in_partition_sidx': sidx}
				potential_positives.append(reldct)
				sents_with_potpos.add(sidx)

	print(f"Total number of relations: {all_num_rels}")
	print(f"Total number of potential positives for partition {partition_key}: {len(potential_positives)};")
	print(f"Total number of sentences with potential positives: {len(sents_with_potpos)}")
	return potential_positives, len(sents_with_potpos)


# TODO: Finished English version, not tested;
def partition_data(args):
	webhose_sents = set()
	clue_sents = set()
	datemngr = DateManager(lang=args.lang)

	# if args.disjoint_window, each key in the clue_time_slices corresponds to a k-day time interval, otherwise corresponds to a specific date
	if args.disjoint_window:
		slice_keys, date2slice = datemngr.setup_dateslices(args.time_interval)
	else:
		slice_keys, date2slice = datemngr.setup_dates(args.time_interval)

	if not os.path.exists(args.int_res_path):
		os.mkdir(args.int_res_path)
	output_fns = {x: os.path.join(args.int_res_path, args.int_res_fn % (args.disjoint_str, x)) for x in slice_keys}
	for k in output_fns:
		ofp = open(output_fns[k], 'w', encoding='utf8')
		ofp.close()

	if args.eg_corpus_fn is not None and not args.preserve_webhose:
		print(f"Reading in EntGraph corpus sentences for deduplication...")
		with open(args.eg_corpus_fn, 'r', encoding='utf8') as wb_fp:
			for line in wb_fp:
				item = json.loads(line)
				webhose_sents.add(item['s'])
		print(f"Finished reading; EntGraph corpus has {len(webhose_sents)} unique sentences;")

	clue_sent_in_webhose_count = 0
	clue_redundant_count = 0

	with open(args.input_fn, 'r', encoding='utf8') as in_fp:
		for lidx, line in enumerate(in_fp):
			if 0 <= args.max_num_lines < lidx:
				break
			if lidx % 100000 == 0:
				print(f"lidx: {lidx}; clue_sent_in_webhose_count: {clue_sent_in_webhose_count}; clue_redundant_count: {clue_redundant_count}")
			# {'s': , 'date': x, 'articleId': x, 'lineId': x, 'rels': [{"r": "(pred::subj::obj::[EE/EG/GE/GG]::0::x::type_subj::type_obj)"}, ...]}
			try:
				item = json.loads(line)
			except json.decoder.JSONDecodeError as e:
				print(f"Skipping line: ``{line}''")
				continue
			if item['s'] in webhose_sents:  # exclude the sentences present in Webhose corpus
				clue_sent_in_webhose_count += 1
				continue
			if item['s'] in clue_sents:
				clue_redundant_count += 1
				continue
			else:
				clue_sents.add(item['s'])
			item_time = item['date']
			try:
				item_time = parse_time(item_time, args.lang)
			except ValueError as e:
				print(lidx)
				print(line)
				raise
			date_str = f"{item_time['year']}-{item_time['month']}-{item_time['day']}"
			cur_slice = date2slice[date_str] if date2slice is not None else f"{date_str}_{args.time_interval}"
			with open(output_fns[cur_slice], 'a', encoding='utf8') as ofp:
				out_line = json.dumps(item, ensure_ascii=False)
				ofp.write(out_line+'\n')
	print(f"Total clue_sent_in_webhose_count: {clue_sent_in_webhose_count}; total clue_redundant_count: {clue_redundant_count}!")

	# if args.store_partitions:  # default to False
	# 	print(f"Writing partitions to {os.path.join(args.int_res_path, args.int_res_fn)}:")
	# 	if not os.path.exists(args.int_res_path):
	# 		os.mkdir(args.int_res_path)
	# 	for k in clue_time_slices:
	# 		with open(os.path.join(args.int_res_path, args.int_res_fn % (args.disjoint_str, k)), 'w', encoding='utf8') as int_fp:
	# 			for item in clue_time_slices[k]:
	# 				out_line = json.dumps(item, ensure_ascii=False)
	# 				int_fp.write(out_line + '\n')
	# 	print(f"Saved.")
	return


def find_potential_positives(args, clue_time_slices=None, accepted_preds=None):

	def read_file_from_key(k):
		res = []
		with open(os.path.join(args.int_res_path, args.int_res_fn % (args.disjoint_str, k)), 'r', encoding='utf8') as fp:
			for line in fp:
				it = json.loads(line)
				res.append(it)
		return res

	datemngr = DateManager(lang=args.lang)

	if accepted_preds is None:
		with open(args.accepted_preds_fn % (args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode), 'r', encoding='utf8') as ap_fp:
			accepted_preds = json.load(ap_fp)
	accepted_preds = set(accepted_preds)

	all_potential_positives = {}
	all_num_sents_with_potpos = {}

	potpos_size = 0

	if clue_time_slices is None:
		assert os.path.isdir(args.int_res_path)
		files = os.listdir(args.int_res_path)
		files.sort()
		_keys = []
		assert args.int_res_fn[-5:] == '.json'
		desired_prefix = args.int_res_fn[:-5] % (args.disjoint_str, '')
		for f in files:
			if f.startswith(desired_prefix):
				_keys.append(f[len(desired_prefix):-5])
	else:
		_keys = list(clue_time_slices.keys())

	for k in _keys:
		# fetch the current bucket
		if args.disjoint_window:
			if clue_time_slices is not None:
				cur_bucket = clue_time_slices[k]
			else:
				cur_bucket = read_file_from_key(k)
		else:
			back_rec_steps = floor((args.time_interval-1)/2)
			fwd_rec_steps = ceil((args.time_interval-1)/2)
			back_date = k
			fwd_date = k
			if clue_time_slices is not None:
				cur_bucket = copy.deepcopy(clue_time_slices[k])
			else:
				cur_bucket = read_file_from_key(k)
			for s in range(back_rec_steps):
				back_date = datemngr.get_prev_date(back_date)
				if back_date is not None:
					if clue_time_slices is not None:
						cur_bucket += clue_time_slices[back_date]
					else:
						cur_bucket += read_file_from_key(back_date)
				else:
					break
			for s in range(fwd_rec_steps):
				fwd_date = datemngr.get_next_date(fwd_date)
				if fwd_date is not None:
					if clue_time_slices is not None:
						cur_bucket += clue_time_slices[fwd_date]
					else:
						cur_bucket += read_file_from_key(fwd_date)
				else:
					break

		# Here the pronouns are excluded.
		cur_partition_accepted_eps = filter_partition_entpairs(cur_bucket, args.slice_entpair_thres,
															   args.slice_entpair_cap, args.ep_filter_mode, k, lang=args.lang)
		cur_partition_accepted_eps = set(cur_partition_accepted_eps)
		slice_potential_positives, num_sent_with_potpos = filter_positive_triples(cur_bucket, accepted_preds, cur_partition_accepted_eps, k, lang=args.lang)
		all_potential_positives[k] = slice_potential_positives
		potpos_size += len(slice_potential_positives)
		all_num_sents_with_potpos[k] = num_sent_with_potpos

	if potpos_size > 1000000:
		potpos_ratio = 1000000 / potpos_size
		print(f"Warning: potential positives size is larger than {1000000}, sampling by ratio {potpos_ratio} ......")
	else:
		potpos_ratio = None

	if args.store_potpos:
		print(f"Writing potential positives to {os.path.join(args.int_res_path, args.potential_pos_fn)}:")
		if not os.path.exists(args.int_res_path):
			os.mkdir(args.int_res_path)
		for k in all_potential_positives:
			with open(os.path.join(args.int_res_path,
								   args.potential_pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap,
															args.total_pred_thres, args.total_pred_cap,
															args.pred_filter_mode, args.ep_filter_mode,
															args.disjoint_str, k)),
					  'w', encoding='utf8') as pot_fp:
				if potpos_ratio is not None:
					all_potential_positives[k] = random.sample(all_potential_positives[k], int(len(all_potential_positives[k]) * potpos_ratio))
				for item in all_potential_positives[k]:
					out_line = json.dumps(item, ensure_ascii=False)
					pot_fp.write(out_line+'\n')
		print(f"Saved.")
		print(f"Writing num_sents_with_potpos info to {os.path.join(args.int_res_path, args.num_sents_with_potpos_fn)}:")
		with open(os.path.join(args.int_res_path, args.num_sents_with_potpos_fn % (
				args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap,
				args.pred_filter_mode, args.ep_filter_mode, args.disjoint_str)), 'w', encoding='utf8') as nswp_fp:
			json.dump(all_num_sents_with_potpos, nswp_fp, indent=4, ensure_ascii=False)
	return all_potential_positives, all_num_sents_with_potpos


def sample_positives(args, all_potential_positives=None, all_num_sents_with_potpos=None):
	if all_potential_positives is None:
		assert all_num_sents_with_potpos is None
		assert os.path.exists(args.int_res_path)
		all_potential_positives = {}
		files = os.listdir(args.int_res_path)
		files.sort()
		assert args.potential_pos_fn[-5:] == '.json'
		desired_prefix = args.potential_pos_fn[:-5] % (args.slice_entpair_thres, args.slice_entpair_cap,
													   args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode,
													   args.ep_filter_mode, args.disjoint_str, '')
		for f in files:
			if f.startswith(desired_prefix):
				k = f[len(desired_prefix):-5]
				all_potential_positives[k] = []
				with open(os.path.join(args.int_res_path, f), 'r', encoding='utf8') as pp_fp:
					for line in pp_fp:
						it = json.loads(line)
						all_potential_positives[k].append(it)
		print(f"Recovered {len(all_potential_positives)} partitions!")

		with open(os.path.join(args.int_res_path, args.num_sents_with_potpos_fn % (
				args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode,
				args.ep_filter_mode, args.disjoint_str)), 'r', encoding='utf8') as fp:
			all_num_sents_with_potpos = json.load(fp)
	else:
		assert all_num_sents_with_potpos is not None

	pronouns = ['我', '你', '他', '她', '它', '他们', '她们', '它们', '有人', '自己', '人', 'I', 'you', 'he', 'she', 'it', 'them',
				'someone', 'somebody', 'one', 'myself', 'yourself', 'yourselves', 'himself', 'herself', 'themselves']
	total_num_potpos = 0
	total_numsents_with_potpos = 0
	for k in all_potential_positives:
		total_num_potpos += len(all_potential_positives[k])
		total_numsents_with_potpos += all_num_sents_with_potpos[k]

	print(f"Total number of potential positives: {total_num_potpos}")
	print(f"Total number of sentences with potential positives: {total_numsents_with_potpos}")
	sample_per_sent = args.sample_size / total_numsents_with_potpos  # this value should always be smaller than 1!
	assert 0 < sample_per_sent < 0.5
	num_samples_alloc = {k: floor(sample_per_sent * all_num_sents_with_potpos[k]) for k in all_num_sents_with_potpos}
	all_sampled_positives = {}
	for k in all_potential_positives:
		part_sampled_positives = {}  # only one relation per sentence may be selected as positive
		while len(part_sampled_positives) < num_samples_alloc[k]:
			cur_spl = random.choice(all_potential_positives[k])
			upred, subj, obj, tsubj, tobj = parse_rel(cur_spl)
			if subj in pronouns or obj in pronouns:  # don't sample it if either subject or object is pronoun
				continue
			if cur_spl['in_partition_sidx'] not in part_sampled_positives:
				part_sampled_positives[cur_spl['in_partition_sidx']] = cur_spl
		all_sampled_positives[k] = list(part_sampled_positives.values())

	# divide samples into dev set and test set such that all positives in each partition may appear in only one subset.
	test_samples = []
	dev_samples = []

	time_slices_reshuffle = list(all_sampled_positives.keys())
	random.shuffle(time_slices_reshuffle)

	for k in time_slices_reshuffle:
		if len(dev_samples) + len(all_sampled_positives[k]) < args.sample_size/2:
			dev_samples += all_sampled_positives[k]
		else:
			test_samples += all_sampled_positives[k]
	print(f"Final dev set positives size: {len(dev_samples)}; final test set positives size: {len(test_samples)}")

	with open(args.pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode, args.ep_filter_mode, args.sample_size, args.disjoint_str, 'dev'), 'w', encoding='utf8') as fp:
		for item in dev_samples:
			out_line = json.dumps(item, ensure_ascii=False)
			fp.write(out_line+'\n')
	with open(args.pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode, args.ep_filter_mode, args.sample_size, args.disjoint_str, 'test'), 'w', encoding='utf8') as fp:
		for item in test_samples:
			out_line = json.dumps(item, ensure_ascii=False)
			fp.write(out_line+'\n')


def remove_skip_preds(fn, lang):
	print(f"Removing skip_preds for positive file: {fn};")
	positives = []
	with open(fn, 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			positives.append(item)

	with open(fn, 'w', encoding='utf8') as fp:
		for item in positives:
			upred, subj, obj, tsubj, tobj = parse_rel(item)
			upred_bow = upred2bow(upred, lang=lang)
			if check_vague(upred_bow) is True:
				continue
			else:
				out_line = json.dumps(item, ensure_ascii=False)
				fp.write(out_line+'\n')


def trim_potential_positives(args):
	datemngr = DateManager(lang=args.lang)
	if args.disjoint_window:
		all_slices, _ = datemngr.setup_dateslices(args.time_interval)
	else:
		all_slices, _ = datemngr.setup_dates(args.time_interval)
	potential_fns = [os.path.join(args.int_res_path,
								  args.potential_pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap,
														   args.total_pred_thres, args.total_pred_cap,
														   args.pred_filter_mode, args.ep_filter_mode,
														   args.disjoint_str, x))
					 for x in all_slices]

	with open(os.path.join(args.int_res_path, args.num_sents_with_potpos_fn %
											  (args.slice_entpair_thres, args.slice_entpair_cap,
											   args.total_pred_thres, args.total_pred_cap,
											   args.pred_filter_mode, args.ep_filter_mode, args.disjoint_str)
						   ), 'r', encoding='utf8') as fp:
		all_num_sents_with_potpos = json.load(fp)

	total_cnt = sum(all_num_sents_with_potpos.values())
	ratio = min(1000000 / total_cnt, 1.0)

	for fn in potential_fns:
		with open(fn, 'r', encoding='utf8') as fp:
			lines = []
			for line in fp:
				assert line[-1] == '\n'
				lines.append(line[:-1])
			curr_samplesize = int(len(lines) * ratio)
			print(f"Trimming potential positives file: {fn}; original size: {len(lines)}; new size: {curr_samplesize}")
			random.shuffle(lines)
			lines = lines[:curr_samplesize]
		with open(fn, 'w', encoding='utf8') as fp:
			for line in lines:
				fp.write(line+'\n')
	print(f"Trimming potential positives done.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='all', type=str, help="Note that mode `all' does not include `partition_data'")
	parser.add_argument('--eg_corpus_fn', default='../typed_triples_tacl.json', type=str)
	parser.add_argument('--input_fn', default='../clue_typed_triples_tacl.json', type=str)
	parser.add_argument('--time_interval', default=3, type=int)
	parser.add_argument('--store_partitions', action='store_true')
	parser.add_argument('--store_potpos', action='store_true')
	parser.add_argument('--disjoint_window', action='store_true')
	parser.add_argument('--preserve_webhose', action='store_true')
	parser.add_argument('--int_res_path', default='./clue_time_slices/', type=str)
	parser.add_argument('--int_res_fn', default="clue_typed_triples_%s_%s.json", type=str)
	parser.add_argument('--accepted_preds_fn', default='clue_accepted_preds_%d_%d_%s.json', type=str)
	parser.add_argument('--potential_pos_fn', default='clue_potential_positives_ep_min%d_max%d_pd_min%d_max%d_%s_%s_%s_%s.json', type=str)
	parser.add_argument('--num_sents_with_potpos_fn', default='clue_num_sents_with_potpos_ep_min%d_max%d_pd_min%d_max%d_%s_%s_%s.json', type=str)
	parser.add_argument('--slice_entpair_thres', default=5, type=int, help='the minimum number of sentences in the current partition in which an entpair needs to be present.')
	parser.add_argument('--slice_entpair_cap', default=100, type=int, help='the maximum number of sentences in the current partition in which an entpair needs to be present.')
	parser.add_argument('--total_pred_thres', default=10, type=int, help='the minumum number of sentences in the whole corpus in which a predicate needs to be present.')
	parser.add_argument('--total_pred_cap', default=0, type=int, help='the maximum number of sentences in the whole corpus in which a predicate could be present: '
																	   'argument should be set for de-biasing the dataset; default value 0 means no capping.')
	parser.add_argument('--pred_filter_mode', default='triple', type=str, help='granularity for counting occurrences when filtering predicates: [triple/sent/doc]')
	parser.add_argument('--ep_filter_mode', default='doc', type=str, help='granularity for counting occurrences when filtering entity pairs: [triple/sent/doc]')
	parser.add_argument('--sample_size', default=40000, type=int, help='total number of samples to sample, half for dev, half for test.')
	parser.add_argument('--max_num_lines', default=-1, type=int, help='Maximum number of lines of clue data to read in, for debugging.')
	parser.add_argument('--pos_fn', default='./clue_positives_ep_min%d_max%d_pd_min%d_max%d_%s_%s_%d_%s_%s.json', type=str)

	parser.add_argument('--lang', default='zh', type=str, help='[zh / en]')
	args = parser.parse_args()
	args.disjoint_str = 'disjoint' if args.disjoint_window else 'sliding'
	assert args.lang in ['zh', 'en']
	assert args.mode in ['all', 'partition_data', 'filter_preds', 'downstream', 'sample', 'update', 'trim']

	if args.mode in ['partition_data']:
		partition_data(args)

	if args.mode in ['filter_preds', 'all']:
		with open(args.input_fn, 'r', encoding='utf8') as in_fp:
			accepted_preds = filter_fullcorpus_predicates(in_fp, args.total_pred_thres, args.total_pred_cap,
														  args.pred_filter_mode, lang=args.lang)
		with open(args.accepted_preds_fn % (args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode), 'w', encoding='utf8') as out_fp:
			json.dump(accepted_preds, out_fp, ensure_ascii=False)
	else:
		accepted_preds = None

	if args.mode in ['downstream', 'all']:
		all_potential_positives, all_num_sents_with_potpos = find_potential_positives(args, None, accepted_preds)
	else:
		all_potential_positives = None
		all_num_sents_with_potpos = None

	if args.mode in ['trim', 'all']:
		trim_potential_positives(args)

	if args.mode in ['sample', 'all']:
		sample_positives(args, all_potential_positives, all_num_sents_with_potpos)
	else:
		pass

	if args.mode == 'update':
		datemngr = DateManager(lang=args.lang)
		dev_sample_fn = args.pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode, args.ep_filter_mode, args.sample_size, args.disjoint_str, 'dev')
		test_sample_fn = args.pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap, args.total_pred_thres, args.total_pred_cap, args.pred_filter_mode, args.ep_filter_mode, args.sample_size, args.disjoint_str, 'test')

		if args.disjoint_window:
			all_slices, _ = datemngr.setup_dateslices(args.time_interval)
		else:
			all_slices, _ = datemngr.setup_dates(args.time_interval)
		potential_fns = [os.path.join(args.int_res_path,
									  args.potential_pos_fn % (args.slice_entpair_thres, args.slice_entpair_cap,
															   args.total_pred_thres, args.total_pred_cap,
															   args.pred_filter_mode, args.ep_filter_mode,
															   args.disjoint_str, x))
						 for x in all_slices]
		remove_skip_preds(dev_sample_fn, lang=args.lang)
		remove_skip_preds(test_sample_fn, lang=args.lang)
		for pfn in potential_fns:
			remove_skip_preds(pfn, lang=args.lang)
	else:
		pass

	# datemngr = DateManager()
	# all_slices = datemngr.setup_dateslices(args.time_interval)
	# print(all_slices)
