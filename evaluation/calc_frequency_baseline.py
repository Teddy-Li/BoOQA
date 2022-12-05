import json
import argparse
import os
import matplotlib.pyplot as plt
from qaeval_utils import parse_rel, rel2concise_str
import sys
sys.path.append('/Users/teddy/eclipse-workspace/entgraph_eval')
sys.path.append('../entgraph_eval')
sys.path.append('../utils/')
import torch
from sklearn.metrics import precision_recall_curve as pr_curve_sklearn
from pytorch_lightning.metrics.functional.classification import precision_recall_curve as pr_curve_pt
from pytorch_lightning.metrics.functional import auc
from qaeval_utils import get_auc


def compute_ss_auc(precisions: torch.FloatTensor, recalls: torch.FloatTensor,
                filter_threshold: float = 0.5) -> torch.FloatTensor:
    xs, ys = [], []
    for p, r in zip(precisions, recalls):
        if p >= filter_threshold:
            xs.append(r)
            ys.append(p)

    return auc(
        torch.cat([x.unsqueeze(0) for x in xs], 0),
        torch.cat([y.unsqueeze(0) for y in ys], 0)
    )


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--sample_fn', type=str, default='nc_final_samples_15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_dev_freqmap.json')
parser.add_argument('--pr_curve_root', type=str, default='./qaeval_freq_bsln_pr_curves/')
parser.add_argument('--cnt_mode', type=str, default='predstr')

args = parser.parse_args()
print(args)
assert args.cnt_mode in ['predstr', 'pred']

all_preds_set = {}

with open('./nc_all_pred_set.json', 'r', encoding='utf8') as fp:
	for lidx, line in enumerate(fp):
		if lidx % 1000000 == 0:
			print(lidx)
		if len(line) < 2:
			continue
		item = json.loads(line)
		if item['type'] not in all_preds_set:
			all_preds_set[item['type']] = {}
		assert item['predstr'] not in all_preds_set[item['type']]
		all_preds_set[item['type']][item['predstr']] = {"p": item['p'], "n": item['n']}


freq_scores = []
gold_labels = []

with open(args.sample_fn, 'r', encoding='utf8') as ifp:
	for lidx, line in enumerate(ifp):
		if len(line) < 2:
			continue
		item = json.loads(line)
		upred, subj, obj, tsubj, tobj = parse_rel(item)
		type_str = f"{tsubj}#{tobj}"
		_, predstr = rel2concise_str(upred, subj, obj, tsubj, tobj, 'en')

		if args.cnt_mode == 'predstr':
			this_freq = all_preds_set[type_str][predstr]['n']
		elif args.cnt_mode == 'pred':
			p_cnt = None
			ps = all_preds_set[type_str][predstr]['p']
			for p, c in ps:
				if p == upred:
					assert p_cnt is None
					p_cnt = c
			if p_cnt is None:
				print(f"cnt missing for pred: {upred};")
				p_cnt = 0
			this_freq = p_cnt
		else:
			raise AssertionError
		assert this_freq > 0
		freq_scores.append((1-1/this_freq))

		if item['label'] is True:
			gold_labels.append(1)
		elif item['label'] is False:
			gold_labels.append(0)
		else:
			raise AssertionError

dataset_bsln_prec = float(sum(gold_labels)) / len(gold_labels)

skl_prec, skl_rec, skl_thres = pr_curve_sklearn(gold_labels, freq_scores)
assert len(skl_prec) == len(skl_rec) and len(skl_prec) == len(skl_thres) + 1
# perhaps report the AUC_BASELINE and the normalized AUC as well!
skl_auc_value = get_auc(skl_prec[1:], skl_rec[1:])
print(f"Hosseini Area under curve: {skl_auc_value};")

assert args.sample_fn[-5:] == '.json'
pr_curve_fp = open(os.path.join(args.pr_curve_root, args.sample_fn[:-5]+'_pr_curve.txt'), 'w', encoding='utf8')

try:
	gold_labels_pt = torch.tensor(gold_labels)
	freq_scores_pt = torch.tensor(freq_scores)
	pt_prec, pt_rec, pt_thres = pr_curve_pt(freq_scores_pt, gold_labels_pt)
	ss_bsln_auc = compute_ss_auc(
		pt_prec, pt_rec,
		filter_threshold=dataset_bsln_prec
	)
	ss_50_auc = compute_ss_auc(
		pt_prec, pt_rec,
		filter_threshold=0.5
	)

	ss_rel_prec = torch.tensor([max(p - dataset_bsln_prec, 0) for p in pt_prec], dtype=torch.float)
	ss_rel_rec = torch.tensor([r for r in pt_rec], dtype=torch.float)
	ss_auc_norm = compute_ss_auc(
		ss_rel_prec, ss_rel_rec,
		filter_threshold=0.0
	)
	ss_auc_norm /= (1 - dataset_bsln_prec)
	print(f"S&S 50 AUC: {ss_50_auc};")
	print(f"S&S bsln AUC: {ss_bsln_auc};")
	print(f"S&S AUC NORM: {ss_auc_norm};")
	print("")
	print("")
	print(f"p\tr\tt")
	for p, r, t in zip(pt_prec, pt_rec, pt_thres):
		pr_curve_fp.write(f"{p}\t{r}\t{t}\n")

except Exception as e:
	print(f"Exception when calculating S&S style AUC!")
	print(e)
	ss_50_auc = None
	ss_bsln_auc = None
	ss_auc_norm = None

