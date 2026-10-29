#!/bin/zsh
# Train + evaluate one TS-MMME variant on the 2024 quarters.
# Usage: run/03_ts_mmme_variant.sh <TI | T_ft | I | TI_ftclip>
#   TI     full model (text + image), frozen encoders            -> Tables 3, 6 (ablations), 7
#   T_ft   text only, fine-tuned BERT_T encoder (frozen inside)  -> Tables 3, 4 (T)
#   I      image only, CLIP image projection, image articles      -> Tables 3, 4 (I)
#   TI_ftclip text + image, fine-tuned BERT_T + CLIP proj.  -> input to gating/make_gated.py (Table 4)
# T_ft / TI_ftclip need Resources/Cache/ft_text_features.pt (data_prep/extract_ft_features.py);
# I / TI_ftclip need Resources/Cache/clip_image_features.pt (data_prep/extract_clip_features.py writes it).
set -e; V=$1; cd "$(dirname "$0")/../ts_mmme"
export SELECT_MACRO_F1=1 VAL_HOLDOUT=1 CLASS_WEIGHTED=1 ANCHOR_TIME=1
unset TEXT_ONLY IMAGE_ONLY FT_TEXT IMG_CLIP_PROJ IMG_ONLY_WITH_IMAGE
MODES=(1)
case $V in
  TI)    MODES=(1 2 3 4 5) ;;
  T_ft)  export TEXT_ONLY=1 FT_TEXT=1 ;;
  I)     export IMAGE_ONLY=1 IMG_CLIP_PROJ=1 IMG_ONLY_WITH_IMAGE=1 ;;
  TI_ftclip) export FT_TEXT=1 IMG_CLIP_PROJ=1; MODES=(1 2 3 4 5) ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
typeset -A NAME; NAME[1]=FULL; NAME[2]=NO_EVOLUTION; NAME[3]=NO_CROSS_ATTENTION; NAME[4]=NO_TEMPORAL; NAME[5]=NO_SPATIAL
OUT=../results/runs/TSMMME_$V; mkdir -p $OUT
# variant-specific memory features, then Graph Transformer + memory bank, then Phase 2
for s in step6_generate_article_text_features step8_generate_memory_text_features Step10_Memory_multimodal_fusion \
         step11_create_temporal_memory_edges step12_build_heterodata step13.1_train_graph_transformer step13.2_generate_zmm; do
  python Scripts/$s.py; done
python Scripts/Phase2_Inference_Model_Training_Pipeline.py
for m in $MODES; do for k in 1 2 3 4; do
  cp Dataset/Experiment/test_2024_Q${k}.json Dataset/Experiment/test_dataset.json
  printf "$m\n2\n" | python Scripts/Phase3_End_to_End_Inference_Pipeline.py > $OUT/${NAME[$m]}_Q${k}.log 2>&1
  cp Results/${NAME[$m]}_predictions.csv $OUT/${NAME[$m]}_Q${k}_predictions.csv
done; done
