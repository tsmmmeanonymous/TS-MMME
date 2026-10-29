#!/bin/bash
# FTT Step 1 (reweighting) on MMM-Trend. Settings copied verbatim from the original
# ICTMCG/FTT-ACL23 reweight_roll_season/run.sh; only data_name/year_type differ.
# Usage: ./run_mmmtrend.sh <data_name> <year_type>   e.g. mmmtrend2023 mmmtrend2023
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
data_name="${1:-mmmtrend}"
year_type="${2:-mmmtrend}"
sentence_transformer_path='imxly/sentence_roberta_wwm_ext'   # original FTT Chinese SBERT
embedding_type='SBERT'
cluster_threshold=0.6
predict_threshold=3
reweight_method='sw'
reweight_threshold=5.0
thres_low=0.5
thres_high=2.0
predict_method='bsr'
A="--data_name ${data_name} --embedding_type ${embedding_type} --cluster_threshold ${cluster_threshold} --predict_method ${predict_method} --predict_threshold ${predict_threshold} --reweight_method ${reweight_method} --reweight_threshold ${reweight_threshold} --thres_low ${thres_low} --thres_high ${thres_high}"
echo "S1 News Representation $(date)";   "$PY" get_embeddings.py $A --sentence_transformer_path ${sentence_transformer_path}
echo "S2 Topic Discovery $(date)";       "$PY" single_pass_cluster.py $A
echo "S3 Forecasting $(date)";           "$PY" predict_freq.py $A --year_type ${year_type}
echo "S4 Adaptation $(date)";            "$PY" weight_score_cal.py $A
echo "STEP1 DONE $(date)"
