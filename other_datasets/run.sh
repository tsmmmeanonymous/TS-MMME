#!/bin/zsh
# Run TS-MMME on one prepared benchmark (see prepare.py) with the same settings as the MMM-TREND run:
# frozen sentence encoder with captions, validation hold-out, selection on validation macro-F1, class-weighted loss.
# Usage: other_datasets/run.sh <name>      Output: results/other_datasets/<name>/
set -e
NAME=$1; REPO="$(cd "$(dirname "$0")/.." && pwd)"; W="$REPO/other_datasets/work/$NAME"; O="$REPO/results/other_datasets/$NAME"
[[ -d $W ]] || { echo "run other_datasets/prepare.py --name $NAME first"; exit 1; }
export SELECT_MACRO_F1=1 VAL_HOLDOUT=1 CLASS_WEIGHTED=1 ANCHOR_TIME=1
mkdir -p $O; cd $W
for s in step00_generate_caption_cache step01_generate_text_embeddings step02_generate_caption_embeddings \
         step14_1_prepare_topic_embeddings step04_load_cached_resources step1_create_graph_nodes step2_create_graph_edges \
         step3_create_temporal_groups step4_create_memory_nodes step5_create_memory_edges \
         step6_generate_article_text_features step7_generate_article_image_features step8_generate_memory_text_features \
         step9_generate_memory_image_features Step10_Memory_multimodal_fusion step11_create_temporal_memory_edges \
         step12_build_heterodata step13.1_train_graph_transformer step13.2_generate_zmm step13_3_train_inference_modules; do
  python Scripts/$s.py > Logs/$s.log 2>&1
done
printf "1\n2\n" | python Scripts/Phase3_End_to_End_Inference_Pipeline.py > Logs/phase3.log 2>&1
cp Results/FULL_predictions.csv Results/FULL_metrics.json $O/
echo "$NAME done -> $O"
