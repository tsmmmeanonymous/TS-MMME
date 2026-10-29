#!/bin/zsh
# All baselines on the same protocol (train <=2023-Q4, test each 2024 quarter). Outputs: $MMMTREND_DATA/bench_results/
set -e; cd "$(dirname "$0")/../baselines"
python bench_bert_year.py 2024          # BERT_T (mBERT, last layer fine-tuned); also saves the encoder used by T_ft / TI_ftclip
python bench_eann.py                    # EANN_T
(cd ftt_reweight && ./run_mmmtrend.sh mmmtrend mmmtrend)   # FTT step 1: original reweighting
python bench_ftt.py                     # FTT_T step 2: weighted BERT, one model per quarter
for m in clip_image blip_image qwen_image clip_vl blip_vl qwen_vl; do python bench_mm.py $m; done   # frozen I / TI baselines
cd ../data_prep
python extract_ft_features.py           # fine-tuned BERT_T features for TS-MMME T_ft / TI_ftclip
python extract_clip_features.py         # CLIP image feature cache for TS-MMME I / TI_ftclip
