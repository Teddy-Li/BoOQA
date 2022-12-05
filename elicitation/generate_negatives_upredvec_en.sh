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
version=$1
global_presence=$2

python -u generate_negatives.py --do_compute_upred_vecs --positives_base nc_positives_%s_%s.json \
--negatives_base nc_negatives_%s_%d_%s_%s.json --potential_positives_path ./nc_time_slices/nc_potential_positives_%s_%s.json \
--potential_negatives_path ./nc_time_slices/nc_potential_negatives_%s_%d_%s_%s.json --time_interval 3 --version "$version" \
--which_positives XXX --global_presence "$global_presence" --wordnet_dir XXX --word2vec_path glove.840B.300d.txt \
--all_triples_path ../news_genC_GG_typed.json --partition_triples_path ./nc_time_slices/nc_typed_triples_%s_%s.json \
--triple_set_path ./nc_all_triple_set.json --pred_set_path ./nc_all_pred_set.json --pred_vectors_path ./nc_triple_vectors.h5 \
--pred_vectors_cache_size 300000 --global_presence_thres 3 --only_negable_pos_flag \
--wn_only_entries_flag --neg_source wordnet --final_samples_base_fn nc_final_samples_%s_%d_%d_%d_%s_%s_%s.json --lang en \
--wsd_model_dir /disk/scratch_big/tli/BERT-WSD/model/bert_base-augmented-batch_size=64-lr=2e-5-max_gloss=4 \
--wsd_batch_size 64

#ls -l
#cp -v ./clue_all_triple_set.json /home/s2063487/QAEval/
#cp -v ./clue_all_pred_set.json /home/s2063487/QAEval/
#cp -v ./clue_triple_vectors.h5 /home/s2063487/QAEval/

echo "Finished!"

