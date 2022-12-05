#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="select_pos"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

mkdir -p /disk/scratch/s2063487

cd ..
cp -r ./QAEval /disk/scratch/s2063487/
cp ./clue_typed_triples_tacl.json /disk/scratch/s2063487
cp ./typed_triples_tacl.json /disk/scratch/s2063487
cd /disk/scratch/s2063487/QAEval || exit 1
pwd

disjoint_window=$1

python -u select_positives.py --mode partition_data --store_partitions --time_interval 3 ${disjoint_window}

cp -r ./clue_time_slices /home/s2063487/QAEval/

echo "Finished!"

