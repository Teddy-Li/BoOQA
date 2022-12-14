#!/bin/bash
#SBATCH --partition=General_Usage
#SBATCH --job-name="gen_neg"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

mkdir -p /disk/scratch/s2063487

version=$1
global_presence_thres=$2
global_presence_cap=$3
balanced_samples_negs_per_pos=$4
neg_source=$5
pos_size_of_final_sample=$6
freq_multiples_cap=$7
flag1=$8  # this flag can be debug


python -u generate_negatives.py --do_final_sampling_freq_map --positives_base nc_positives_%s_%s.json \
--negatives_base nc_negatives_%s_%d_%d_%s_%s.json \
--potential_positives_path ../nc_time_slices/nc_potential_positives_%s_%s.json \
--potential_negatives_path ../nc_time_slices/nc_potential_negatives_%s_%d_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives potential --global_presence pred --wordnet_dir XXX \
--wsd_model_dir /Users/teddy/PycharmProjects/BERT-WSD/model/bert_base-augmented-batch_size=64-lr=2e-5-max_gloss=4 \
--word2vec_path ../nc_aux_data/glove.840B.300d.txt --all_triples_path ../../news_genC_GG_typed.json --wsd_batch_size 32 \
--partition_triples_path ../nc_time_slices/nc_typed_triples_%s_%s.json --triple_set_path ../nc_inter_data/nc_all_triple_set.json \
--pred_set_path ../nc_inter_data/nc_all_pred_set.json --pred_vectors_path ../nc_inter_data/nc_triple_vectors.h5 --pred_vectors_cache_size 300000 \
--global_presence_thres "$global_presence_thres" --global_presence_cap "$global_presence_cap" \
--max_num_posi_collected_per_partition 30000 --balanced_samples_negs_per_pos "$balanced_samples_negs_per_pos" \
--only_negable_pos_flag --wn_then_w2v_flag --wn_only_entries_flag --neg_source "$neg_source" \
--pos_size_of_final_sample "$pos_size_of_final_sample" --freq_multiples_cap "$freq_multiples_cap" \
--final_samples_base_fn nc_final_samples_%s_%d_%d_%d_%d_%s_%s_%.1f_freqmap_%s.json --lang en ${flag1}

echo "Finished!"

