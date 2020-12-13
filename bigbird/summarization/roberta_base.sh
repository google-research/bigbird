#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/summarization/run_summarization.py \
  --data_dir="tfds://scientific_papers/pubmed" \
  --output_dir="$GCP_EXP_BUCKET"summarization/pubmed \
  --attention_type=block_sparse \
  --couple_encoder_decoder=True \
  --max_encoder_length=3072 \
  --max_decoder_length=256 \
  --num_attention_heads=12 \
  --num_hidden_layers=12 \
  --hidden_size=768 \
  --intermediate_size=3072 \
  --block_size=64 \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --do_train=True \
  --do_eval=False \
  --use_tpu=True \
  --tpu_name=bigbird \
  --tpu_zone=europe-west4-a \
  --gcp_project="$GCP_PROJECT_NAME" \
  --num_tpu_cores=64 \
  --init_checkpoint=gs://bigbird-transformer/pretrain/bigbr_base/model.ckpt-0
