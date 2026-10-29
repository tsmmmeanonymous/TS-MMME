#!/bin/zsh
# Phase 1: captions, embeddings, heterogeneous graph, memory nodes, Graph Transformer, memory bank (run once).
set -e; cd "$(dirname "$0")/../ts_mmme"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
python Scripts/Phase1_Offline_Graph_Construction_Pipeline.py
