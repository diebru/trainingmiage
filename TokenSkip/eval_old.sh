#!/bin/bash
BENCHMARK="gsm8k"
MODEL_SIZE="8b"
MODEL_TYPE="llama3"
DATA_TYPE="test"

COMPRESSION_RATIO=0.7

# 1. Puntiamo al modello base ORIGINALE
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

# 2. Puntiamo all'adattatore LoRA appena addestrato 
# (Attenzione al percorso: saliamo di una cartella e entriamo in LlamaFactory)
ADAPTER_PATH="../LlamaFactory/lora_saves/LLaMA-3.1-8B-Instruct/lora/tokenskip_gsm8k"

# 3. Output dinamico (ho aggiunto "adapter_test" per distinguerlo)
OUPTUT_DIR="outputs/LLaMA-3.1-8B-Instruct/${BENCHMARK}/adapter_test_ratio_${COMPRESSION_RATIO}/"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=512
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

# Comando di avvio: aggiungiamo --use_adapter e --adapter-path
CUDA_VISIBLE_DEVICES=0 python ./evaluation.py \
    --output-dir ${OUPTUT_DIR} \
    --model-path ${MODEL_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --model-size ${MODEL_SIZE} \
    --model-type ${MODEL_TYPE} \
    --data-type ${DATA_TYPE} \
    --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
    --benchmark ${BENCHMARK} \
    --use_vllm \
    --compression_ratio ${COMPRESSION_RATIO} \
    --use_adapter \
    --adapter-path ${ADAPTER_PATH}