#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/classifier/run_classifier.py \
  --data_dir="tfds://imdb_reviews/plain_text" \
  --output_dir="$GCP_EXP_BUCKET"classifier/imdb \
  --attention_type=block_sparse \
  --max_encoder_length=4096 \
  --num_attention_heads=12 \
  --num_hidden_layers=12 \
  --hidden_size=768 \
  --intermediate_size=3072 \
  --block_size=64 \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --do_train=True \
  --do_eval=True \
  --use_tpu=True \
  --tpu_name=bigbird \
  --tpu_zone=europe-west4-a \
  --gcp_project="$GCP_PROJECT_NAME" \
  --num_tpu_cores=32 \
  --init_checkpoint=gs://bigbird-transformer/pretrain/bigbr_base/model.ckpt-0
