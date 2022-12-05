#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="select_pos"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

#mkdir -p /disk/scratch/s2063487

pred_thres=$1
pred_cap=$2
entpair_thres=$3  # here the slice thresholds are stricter than total count in the entire corpus; caps looser though.
entpair_cap=$4
pred_filter_mode=$5
ep_filter_mode=$6
sample_size=$7
mode=$8
disjoint_window=$9

python -u select_positives.py --mode "$mode" --eg_corpus_fn ../../entGraph/news_gen8_p.json --input_fn ../../news_genC_GG_typed.json \
--time_interval 3 --store_partitions --store_potpos "${disjoint_window}" --int_res_path ../nc_time_slices \
--int_res_fn nc_typed_triples_%s_%s.json --accepted_preds_fn nc_accepted_preds_%d_%d_%s.json \
--potential_pos_fn nc_potential_positives_ep_min%d_max%d_pd_min%d_max%d_%s_%s_%s_%s.json \
--num_sents_with_potpos_fn nc_num_sents_with_potpos_ep_min%d_max%d_pd_min%d_max%d_%s_%s_%s.json \
--slice_entpair_thres "$entpair_thres" --slice_entpair_cap "${entpair_cap}" --total_pred_thres "$pred_thres" \
--total_pred_cap "$pred_cap" --pred_filter_mode "$pred_filter_mode" --ep_filter_mode "$ep_filter_mode" \
--sample_size "$sample_size" --pos_fn ../nc_inter_data/placeholder_%d_%d_%d_%d_%s_%s_%d_%s_%s.json --lang en

echo "Finished!"