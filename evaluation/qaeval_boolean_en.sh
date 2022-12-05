#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="QA_eval"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


eval_set=$1
version=$2
bert_dir=$3
eval_method=$4
eg_root=$5
eg_name=$6
eg_suff=$7
eg_feat_idx=$8
device_name=$9
# These flags below can include ``debug'', `no_ref_cache'', ``data_parallel'' and ``no_triple_cache''
flag1=${10}
flag2=${11}
flag3=${12}

pwd
python -u qaeval_chinese_scripts.py --eval_set "$eval_set" --version "$version" \
--fpath_base ../nc_data/nc_final_samples_%s_%s.json --sliced_triples_dir ../nc_time_slices/ --slicing_method disjoint \
--time_interval 3 --sliced_triples_base_fn nc_typed_triples_%s_%s.json --eval_mode boolean \
--eval_method "$eval_method" --eg_root "${eg_root}" --eg_name "$eg_name" --eg_suff "$eg_suff" --eg_feat_idx "$eg_feat_idx" \
--max_spansize 64 --result_dir ./results/qaeval_en_results_prd/%s_%s/ --backupAvg --device_name "$device_name" \
--min_graphsize 20480 --max_context_size 3200 --bert_dir "${bert_dir}" --batch_size 16 \
--refs_cache_dir ./cache_dir_en/refs_%s.json --triples_cache_dir ./cache_dir_en/triples_%s.json \
--tfidf_path ../aux_data/nc_doc_db-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz \
--articleIds_dict_path ../aux_data/nc_articleIds_by_partition.json --num_refs_bert1 5 --lang en \
--all_preds_set_path ../nc_inter_data/nc_all_pred_set.json --threshold_samestr 1 ${flag1} ${flag2} ${flag3}

#--wh_fpath_base ./nc_wh_final_samples_%s_%s.json \
#--negi_fpath_base ./nc_negi_final_samples_%s_%s.json \