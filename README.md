# BoOQA Datasets

This repository hosts the BoOQA dataset, a boolean open-domain QA dataset proposed in the Findings of EMNLP paper 
[Language Models are Poor Learners of Directional Inference](https://arxiv.org/abs/2210.04695). 

If you wish to replicate the results in the paper, or use the dataset for evaluating your own models, please follow the instructions below:

1. Download the [Chinese WordNet](https://bond-lab.github.io/cow/) and [English](https://nlp.stanford.edu/projects/glove/) / [Chinese](https://github.com/Embedding/Chinese-Word-Vectors) word embeddings from their respective pages;
2. For English, download the data package from [here](); for Chinese, download the data from [here](https://uoe-my.sharepoint.com/:u:/g/personal/s2063487_ed_ac_uk/EcAnyjRD8K5KoFo6qaxJ8xYBE-ZwBWEIyOy5ow8DLHNZOg?e=iY6jYh);
3. Unzip the downloaded files and place the unzipped folders in the root directory;
4. Run `pip install -r requirements.txt` to install the required packages;
5. Follow instructions in the `evaluation` section below to evaluate your models.

Otherwise, if you wish to build your own dataset, please first refer to the `elicitation` section below, 
and then follow the instructions in the `evaluation` section to evaluate models with your dataset.

## Elicitation

### Select Positives
Use `select_positives_en_generic.sh` or `select_positives_zh_generic.sh`
1. Partision the news corpora with: 
    * EN: `nohup bash select_positives_en_generic.sh en nc_time_slices nc_typed_triples_%s_%s.json ../../entGraph_NS/news_gen8_p.json ../../news_genC_GG.json --disjoint_window`;
    * ZH: `nohup bash select_positives_zh_generic.sh zh clue_time_slices clue_typed_triples_%s_%s.json ../../typed_triples_tacl.json ../../../clue_typed_triples_tacl.json --disjoint_window`;
2. Select positives with:
   * EN:  `nohup bash select_positives_en_generic.sh 30 3000 15 30 triple doc 40000 all "--disjoint_window" > 
   ./nc_logdir/logpos_all_30_3000_15_triple_doc_40000_disjoint.log &`
   * ZH: `nohup bash select_positives_zh_generic.sh 30 0 15 30 triple doc 40000 all "--disjoint_window" > 
   ./nc_logdir/logpos_all_30_0_15_triple_doc_40000_disjoint.log &`
* EN (ep_freqband): 
   * `nohup bash select_positives_en_generic.sh 30 0 MIN_FREQ MAX_FREQ triple sent 40000 downstream "--disjoint_window" > ./nc_logdir/logpos_downstream_30_0_MINFREQ_MAXFREQ_triple_sent_40000_disjoint.log &`
   * `nohup bash select_positives_en_generic.sh 30 0 MIN_FREQ MAX_FREQ triple sent 40000 trim "--disjoint_window" > ./nc_logdir/logpos_trim_30_0_MINFREQ_MAXFREQ_triple_sent_40000_disjoint.log &`

### Generate Negatives
* EN
    1.1 Loading the predicate sets (invariant to different thresholds): `nohup bash generate_negatives_loading_en.sh --global_triple_absence_flag > ./nc_logdir/neg_loading_tripleabsence.log &`
    1.2 Compute the most likely synsets from all WordNet Synsets including this lemma (invariant to different thresholds): 
    `CUDA_VISIBLE_DEVICES="1" nohup bash generate_negatives_synsets.sh ep_min2_max5_pd_min30_max0_triple_sent_disjoint potential pred > ./nc_logdir/logneg_synset_ep_min2_max5_pd_min30_max0_triple_sent_disjoint.log &`
  (2.) Loading vector representations of untyped predicates (invariant to different thresholds): `nohup bash generate_negatives_upredvec_en.sh 
    15_30_3000_triple_doc_40000_disjoint pred > ./nc_logdir/logneg_upredvec_30_3000_15_triple_doc_40000_disjoint.log &`
    3. Generating WordNet Negatives: `nohup bash generate_negatives_wordnet_en.sh ep_min2_max0_pd_min30_max0_triple_sent_disjoint potential pred 30 0 > ./nc_logdir/logneg_wordnet_ep_min2_max0_pd_min30_max0_triple_sent_disjoint_30_0.log &`
    4. Generating Word2Vec Negatives: NOT IMPLEMENTED
    5. Sampling: `nohup bash generate_negatives_sampling_freqmap_en.sh ep_min2_max5_pd_min30_max0_triple_sent_disjoint 30 0 2 wordnet 40000 0 > ./nc_logdir/logneg_sample40k_ep_min2_max5_pd_min30_max0_triple_sent_disjoint_30_0.log &`

* ZH
    1. Loading the predicate sets (invariant to different thresholds): ``
    2. Loading vector representations of untyped predicates (also invariant to different thresholds): ``
    3. Generating WordNet Negatives: ``
    4. Generating Word2Vec Negatives: ``
    5. Sampling: ``

## Evaluation
* EN
    1. Go to entgraph_eval/evaluation
    2. Do the followings according to method
        1. BERT1A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert1A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 --exclusive=user qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap YOUR_BERT_PATH bert1A - - - 4 "cuda:0"`
        2. BERT2A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert2A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap YOUR_BERT_PATH bert2A - - - 4 "cuda:0"`
        3. BERT3A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert3A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap YOUR_BERT_PATH bert3A - - - 4 "cuda:0"`
        4. S&S base full LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_en_levyholt_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/levyholt_en_dirp/levyholt_en_dirp_29/checkpointepoch=4.ckpt "cuda:0"`
        5. S&S base dir LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_endir_levyholt_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/endir_levyholt/endir_levyholt_63/checkpointepoch=4.ckpt "cuda:0"`
        6. S&S base sym LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_ensym_levyholt_test.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh test 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/ensym_all_levyholt/ensym_all_levyholt_41/checkpointepoch=4.ckpt "cuda:0"`
        7. S&S large full LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_en_levyholt_large_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/levyholt_en_large/levyholt_en_large_57/checkpointepoch=4.ckpt "cuda:0"`
        8. S&S large dir LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_endir_levyholt_large_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/endir_levyholt_large/endir_levyholt_large_36/checkpointepoch=4.ckpt "cuda:0"`
        9. S&S large sym LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_ensym_levyholt_large_dev.log -p PGR-Ubuntu --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/ensym_all_levyholt_large/ensym_all_levyholt_large_51/checkpointepoch=3.ckpt "cuda:0"`
        10. EGEN BInc: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg global_graphs _gsim.txt 1 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_global_backoff2predstr_dev.log &`
        11. EGEN CNCE: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg ../../EG_bucket/entgraphs_contextual_5e-4_1e-2_se_aord_self_top_fill_100_bsz512_alpha_.5 _tPropC_i4_1.5_.3_0.00005_thr0.00005.txt 1 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_contextual_global_orig_backoff2predstr_dev.log &`
        12. EGEN EGT2: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg EGT2_global_graphs _gsim.txt 0 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_EGT2_global_backoff2predstr_dev.log &`

[//]: # (        1. BERT1A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert1A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 --exclusive=user qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap bert1A - - 4 "cuda:0"`)

[//]: # (        2. BERT2A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert2A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap bert2A - - 4 "cuda:0"`)

[//]: # (        3. BERT3A: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_bert3A_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap bert3A - - 4 "cuda:0"`)

[//]: # (        4. S&S base full LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_en_levyholt_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/levyholt_en_dirp/levyholt_en_dirp_29/checkpointepoch=4.ckpt "cuda:0"`)

[//]: # (        5. S&S base dir LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_endir_levyholt_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/endir_levyholt/endir_levyholt_63/checkpointepoch=4.ckpt "cuda:0"`)

[//]: # (        6. S&S base sym LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_ensym_levyholt_test.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh test 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/ensym_all_levyholt/ensym_all_levyholt_41/checkpointepoch=4.ckpt "cuda:0"`)

[//]: # (        7. S&S large full LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_en_levyholt_large_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/levyholt_en_large/levyholt_en_large_57/checkpointepoch=4.ckpt "cuda:0"`)

[//]: # (        8. S&S large dir LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_endir_levyholt_large_dev.log -p PGR-Standard --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/endir_levyholt_large/endir_levyholt_large_36/checkpointepoch=4.ckpt "cuda:0"`)

[//]: # (        9. S&S large sym LevyHolt: `sbatch -o ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_freqmap_40000_ss_ensym_levyholt_large_dev.log -p PGR-Ubuntu --exclude damnii02,damnii03,damnii06,damnii08,damnii09 qaeval_boolean_en_ss.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_enr/ensym_all_levyholt_large/ensym_all_levyholt_large_51/checkpointepoch=3.ckpt "cuda:0"`)

[//]: # (        10. EGEN BInc: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg global_graphs _gsim.txt 1 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_global_backoff2predstr_dev.log &`)

[//]: # (        11. EGEN CNCE: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg ../../EG_bucket/entgraphs_contextual_5e-4_1e-2_se_aord_self_top_fill_100_bsz512_alpha_.5 _tPropC_i4_1.5_.3_0.00005_thr0.00005.txt 1 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_contextual_global_orig_backoff2predstr_dev.log &`)

[//]: # (        12. EGEN EGT2: `nohup bash qaeval_boolean_en.sh dev 15_30_0_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_freqmap eg EGT2_global_graphs _gsim.txt 0 "cpu" --backoff_to_predstr > ./qaeval_logs_en/qaeval_bool_wn_only_negi30_0_0_40000_freqmap_ns_EGT2_global_backoff2predstr_dev.log &`)

* ZH
    1. Go to entgraph_eval/evaluation
    2. Do the followings according to method
        1. BERT1A: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_bert1A_dev.log -p PGR-Standard --exclusive=user qaeval_boolean_zh.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap bert1A - - 4 "cuda:0"`
        2. BERT2A: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_bert2A_dev.log -p PGR-Standard qaeval_boolean_zh.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap bert2A - - 4 "cuda:0"`
        3. BERT3A: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_bert3A_dev.log -p PGR-Standard qaeval_boolean_zh.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap bert3A - - 4 "cuda:0"`
        4. S&S base full LevyHolt: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_ss_zhraw_levyholt_dev.log -p PGR-Standard qaeval_boolean_zh_ss.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_xlm/levyholt_raw_raw/levyholt_raw_raw_63/checkpointepoch=4.ckpt "cuda:0"`
        5. S&S base dir LevyHolt: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_ss_zhraw_dir_levyholt_dev.log -p PGR-Standard qaeval_boolean_zh_ss.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_xlm/levyholt_zhdir_raw/levyholt_zhdir_raw_5/checkpointepoch=1.ckpt "cuda:0"`
        6. S&S base sym LevyHolt: `sbatch -o ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_freqmap_40000_ss_zhraw_symall_levyholt_dev.log -p PGR-Standard qaeval_boolean_zh_ss.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap /home/s2063487/multilingual-lexical-inference/lm-lexical-inference/checkpoints_xlm/levyholt_zhsym_all_raw/levyholt_zhsym_all_raw_36/checkpointepoch=3.ckpt "cuda:0"`
        7. EGZH: `nohup bash qaeval_boolean_zh.sh dev 15_30_triple_doc_disjoint_30_0_40000_2_lexic_wordnet_10.0_freqmap eg ../../entGraph_3/typedEntGrDir_Chinese2_2 _binc_2_1e-4_1e-2.txt 1 "cpu" > ./qaeval_logs_zh/qaeval_bool_wn_only_negi30_0_10_40000_freqmap_wh_binc_dev.log &`


## Cite Us

If you use our code or data, please cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2210.04695,
  doi = {10.48550/ARXIV.2210.04695},
  
  url = {https://arxiv.org/abs/2210.04695},
  
  author = {Li, Tianyi and Hosseini, Mohammad Javad and Weber, Sabine and Steedman, Mark},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Language Models Are Poor Learners of Directional Inference},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```