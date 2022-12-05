#!/bin/bash
#SBATCH --partition=General_Usage
#SBATCH --job-name="gen_neg"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

mkdir -p /disk/scratch/s2063487

version=$1
which_positives=$2
global_presence=$3
global_presence_thres=$4
global_presence_cap=$5
# vnonly=$5 # the ``vnonly'' flag is futile for English, so it has been commented out.
# these flags below can be: [first_only / allow_prep_mismatch_replacements / allow_backoff_wsd]
flag1=$6
flag2=$7
flag3=$8

python -u generate_negatives.py --do_wordnet --positives_base nc_positives_%s_%s.json --negatives_base nc_negatives_%s_%d_%d_%s_%s.json \
--potential_positives_path ../nc_time_slices/nc_potential_positives_%s_%s.json \
--potential_negatives_path ../nc_time_slices/nc_potential_negatives_%s_%d_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives "$which_positives" --global_presence "$global_presence" --wordnet_dir XXX \
--wsd_model_dir /Users/teddy/PycharmProjects/BERT-WSD/model/bert_base-augmented-batch_size=64-lr=2e-5-max_gloss=4 \
--word2vec_path ../nc_aux_data/glove.840B.300d.txt --all_triples_path ../../news_genC_GG_typed.json \
--partition_triples_path ../nc_time_slices/nc_typed_triples_%s_%s.json --triple_set_path ../nc_inter_data/nc_all_triple_set.json \
--pred_set_path ../nc_inter_data/nc_all_pred_set.json --pred_vectors_path ../nc_inter_data/nc_triple_vectors.h5 --pred_vectors_cache_size 300000 \
--global_presence_thres "$global_presence_thres" --global_presence_cap "$global_presence_cap" \
--max_num_posi_collected_per_partition 30000 --lang en ${flag1} ${flag2} ${flag3}

echo "Finished!"

