#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="gen_neg"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

version=$1
global_presence=$2

python -u generate_negatives.py --do_load_triples --lang zh --version "$version" \
--global_presence "$global_presence" --all_triples_path ../entGraph_3/clue_typed_triples_tacl.json \
--triple_set_path ../clue_inter_data/clue_all_triple_set.json --pred_set_path ../clue_inter_data/clue_all_pred_set.json



echo "Finished!"

