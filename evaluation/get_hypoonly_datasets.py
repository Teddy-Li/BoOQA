import argparse
import json
import sys
sys.path.append('../utils/')
from qaeval_utils import parse_rel, upred2bow, reconstruct_sent_from_rel


parser = argparse.ArgumentParser()
parser.add_argument('--input_fn', type=str, default='../nc_data/nc_final_samples_15_30_triple_doc_disjoint_5_40000_2_lexic_wordnet_%s.json')
parser.add_argument('--train_num_posis', type=int, default=1600)  #default=1600)
parser.add_argument('--dev_num_posis', type=int, default=400)  #default=400)
parser.add_argument('--output_fn', type=str, default='/Users/teddy/PycharmProjects/multilingual-lexical-inference/datasets/data_qaeval_15_30_5/hypoonly_typearg_lhsize/%s.txt')
parser.add_argument('--arguments', type=str, default='type')

args = parser.parse_args()
assert args.arguments in ['name', 'type']

train_ofp = open(args.output_fn % 'train', 'w', encoding='utf8')
dev_ofp = open(args.output_fn % 'dev', 'w', encoding='utf8')
test_ofp = open(args.output_fn % 'test', 'w', encoding='utf8')

in_dev_num_posis = 0
with open(args.input_fn % 'dev', 'r', encoding='utf8') as dev_ifp:
	for line in dev_ifp:
		if len(line) < 2:
			continue

		item = json.loads(line)
		upred, subj, obj, tsubj, tobj = parse_rel(item)
		upred_list = upred2bow(upred, lang='en')
		upred_surface_form = ' '.join(upred_list)
		if args.arguments == 'name':
			reconstructed_sent = ' '.join([subj, upred_surface_form, obj])
		elif args.arguments == 'type':
			reconstructed_sent = ' '.join([tsubj, upred_surface_form, tobj])
		else:
			raise AssertionError
		if item['label'] is True:
			label_str = 'True'
			in_dev_num_posis += 1
		elif item['label'] is False:
			label_str = 'False'
		else:
			raise AssertionError
		out_line = f"{reconstructed_sent},,\ttrue,,\t{label_str}\tEN\n"
		if in_dev_num_posis <= args.train_num_posis:
			train_ofp.write(out_line)
		elif in_dev_num_posis <= args.train_num_posis + args.dev_num_posis:
			dev_ofp.write(out_line)
		else:
			pass

with open(args.input_fn % 'test', 'r', encoding='utf8') as test_ifp:
	for line in test_ifp:
		if len(line) < 2:
			continue
		item = json.loads(line)
		upred, subj, obj, tsubj, tobj = parse_rel(item)
		upred_list = upred2bow(upred, lang='en')
		upred_surface_form = ' '.join(upred_list)
		if args.arguments == 'name':
			reconstructed_sent = ' '.join([subj, upred_surface_form, obj])
		elif args.arguments == 'type':
			reconstructed_sent = ' '.join([tsubj, upred_surface_form, tobj])
		else:
			raise AssertionError
		if item['label'] is True:
			label_str = 'True'
		elif item['label'] is False:
			label_str = 'False'
		else:
			raise AssertionError
		out_line = f"{reconstructed_sent},,\ttrue,,\t{label_str}\tEN\n"
		test_ofp.write(out_line)


train_ofp.close()
dev_ofp.close()
test_ofp.close()

print(f"Finished!")
