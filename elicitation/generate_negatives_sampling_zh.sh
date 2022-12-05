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


python -u generate_negatives.py --do_final_sampling --positives_base clue_positives_%s_%s.json \
--negatives_base clue_negatives_%s_%d_%d_%s_%s.json \
--potential_positives_path ../clue_time_slices/clue_potential_positives_%s_%s.json \
--potential_negatives_path ../clue_time_slices/clue_potential_negatives_%s_%d_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives potential --global_presence pred --wordnet_dir XXX --wsd_model_dir XXX \
--word2vec_path ../clue_aux_data/sgns.merge.char --all_triples_path ../../entGraph_3/clue_typed_triples_tacl.json \
--partition_triples_path ../clue_time_slices/clue_typed_triples_%s_%s.json --triple_set_path ../clue_inter_data/clue_all_triple_set.json \
--pred_set_path ../clue_inter_data/clue_all_pred_set.json --pred_vectors_path ../clue_inter_data/clue_triple_vectors.h5 --pred_vectors_cache_size 300000 \
--global_presence_thres "$global_presence_thres" --global_presence_cap "$global_presence_cap" \
--max_num_posi_collected_per_partition 40000 --balanced_samples_negs_per_pos "$balanced_samples_negs_per_pos" \
--only_negable_pos_flag --wn_then_w2v_flag --wn_only_entries_flag --neg_source "$neg_source" \
--pos_size_of_final_sample "$pos_size_of_final_sample" --freq_multiples_cap "$freq_multiples_cap" \
--final_samples_base_fn clue_final_samples_%s_%d_%d_%d_%d_%s_%s_%.1f_%s.json --lang zh ${flag1}


echo "Finished!"

