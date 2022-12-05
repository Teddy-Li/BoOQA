#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="select_pos"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

#mkdir -p /disk/scratch/s2063487

#cd ..
#cp -r ./QAEval /disk/scratch/s2063487/
#cp ./clue_typed_triples_tacl.json /disk/scratch/s2063487
#cp ./typed_triples_tacl.json /disk/scratch/s2063487
#cd /disk/scratch/s2063487/QAEval || exit 1
#pwd


lang=$1
int_root=$2  # root to intermediate results (the time-sliced corpus), e.g. ./clue_time_slices/
int_fn=$3  # filenames of intermediate results (the time-sliced corpus), e.g. clue_typed_triples_%s_%s.json
eg_corpus_fn=$4  # the path to the corpus used for EG construction, to be excluded from evaluation data
news_fn=$5  # the path to the corpus used for construction of evaluation data.
disjoint_window=$6

python -u select_positives.py --mode partition_data --store_partitions --time_interval 3 --lang "${lang}" \
--int_res_path "${int_root}" --int_res_fn "${int_fn}" \
--eg_corpus_fn "${eg_corpus_fn}" --input_fn "${news_fn}" ${disjoint_window}

#cp -r ./clue_time_slices /home/s2063487/QAEval/

echo "Finished!"

