#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="gen_neg"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

#mkdir -p /disk/scratch/s2063487
#
#cd ..
#cp -rv ./QAEval /disk/scratch/s2063487/
##cp -v ./clue_typed_triples_tacl.json /disk/scratch/s2063487
##cp -v ./typed_triples_tacl.json /disk/scratch/s2063487
##cp -rv ./QAEval/generate_negatives* /disk/scratch/s2063487/QAEval/
##cp -rv ./QAEval/select_positives* /disk/scratch/s2063487/QAEval/
##cp -v ./QAEval/qaeval_utils.py /disk/scratch/s2063487/QAEval/
#cd /disk/scratch/s2063487/QAEval || exit 1
#pwd

global_presence=$1
flag1=$2  # could be ``--global_triple_absence_flag''

python -u generate_negatives.py --do_load_triples --all_triples_path ../../news_genC_GG_typed.json \
--global_presence "${global_presence}" --triple_set_path ../nc_inter_data/nc_all_triple_set.json \
--pred_set_path ../nc_inter_data/nc_all_pred_set.json --lang en ${flag1}

#ls -l
#cp -v ./clue_all_triple_set.json /home/s2063487/QAEval/
#cp -v ./clue_all_pred_set.json /home/s2063487/QAEval/
#cp -v ./clue_triple_vectors.h5 /home/s2063487/QAEval/

echo "Finished!"

