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
vnonly=$6
firstonly=$7

python -u generate_negatives.py --do_wordnet --positives_base clue_positives_%s_%s.json --negatives_base clue_negatives_%s_%d_%d_%s_%s.json \
--potential_positives_path ./clue_time_slices/clue_potential_positives_%s_%s.json \
--potential_negatives_path ./clue_time_slices/clue_potential_negatives_%s_%d_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives "$which_positives" --global_presence "$global_presence" --wordnet_dir ./wn-cmn-lmf.xml \
--wsd_model_dir XXX --word2vec_path ./sgns.merge.char --all_triples_path ../entGraph_3/clue_typed_triples_tacl.json \
--partition_triples_path ./clue_time_slices/clue_typed_triples_%s_%s.json --triple_set_path ./clue_all_triple_set.json \
--pred_set_path ./clue_all_pred_set.json --pred_vectors_path ./clue_triple_vectors.h5 --pred_vectors_cache_size 300000 \
--global_presence_thres "$global_presence_thres" --global_presence_cap "$global_presence_cap" \
--max_num_posi_collected_per_partition 40000 --lang zh ${vnonly} ${firstonly}

echo "Finished!"

