#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="select_pos"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

#mkdir -p /disk/scratch/s2063487

pred_thres=$1
pred_cap=$2
entpair_thres=$3
entpair_cap=$4
pred_filter_mode=$4
ep_filter_mode=$5
sample_size=$6
mode=$7
disjoint_window=$8

python -u select_positives.py --mode "$mode" --eg_corpus_fn ../../entGraph_3/typed_triples_tacl.json \
--input_fn ../../entGraph_3/clue_typed_triples_tacl.json --time_interval 3 --store_partitions --store_potpos \
"${disjoint_window}" --int_res_path ../clue_time_slices/ --int_res_fn clue_typed_triples_%s_%s.json \
--accepted_preds_fn clue_accepted_preds_%d_%d_%s.json \
--potential_pos_fn clue_potential_positives_%d_%d_%d_%s_%s_%s_%s.json \
--num_sents_with_potpos_fn clue_num_sents_with_potpos_%d_%d_%d_%s_%s_%s.json \
--slice_entpair_thres "$entpair_thres" --slice_entpair_cap "${entpair_cap}" --total_pred_thres "$pred_thres" \
--total_pred_cap "$pred_cap" --pred_filter_mode "$pred_filter_mode" --ep_filter_mode "$ep_filter_mode" \
--sample_size "$sample_size" --pos_fn ../clue_inter_data/placeholder_%d_%d_%d_%s_%s_%d_%s_%s.json --lang zh
echo "Finished!"