#!/bin/bash
#SBATCH --partition=General_Usage
#SBATCH --job-name="gen_neg"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

mkdir -p /disk/scratch/s2063487

version=$1
which_positives=$2
global_presence=$3

python -u generate_negatives.py --do_compute_posi_synsets --positives_base nc_positives_%s_%s.json \
--negatives_base nc_negatives_%s_%d_%d_%s_%s.json --potential_positives_path ../nc_time_slices/nc_potential_positives_%s_%s.json \
--potential_negatives_path ../nc_time_slices/nc_potential_negatives_%s_%d_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives "$which_positives" --global_presence "$global_presence" --word2vec_path ../nc_aux_data/glove.840B.300d.txt \
--all_triples_path ../../news_genC_GG_typed.json --partition_triples_path ../nc_time_slices/nc_typed_triples_%s_%s.json \
--pred_vectors_cache_size 300000 --only_negable_pos_flag --wn_only_entries_flag --lang en --verbose \
--wsd_model_dir /disk/scratch_big/tli/BERT-WSD/model/bert_base-augmented-batch_size=64-lr=2e-5-max_gloss=4 --wsd_batch_size 64

echo "Finished!"

