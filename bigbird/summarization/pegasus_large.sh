#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/summarization/run_summarization.py \
  --data_dir="tfds://scientific_papers/pubmed" \
  --output_dir="$GCP_EXP_BUCKET"summarization/pubmed \
  --attention_type=block_sparse \
  --couple_encoder_decoder=False \
  --max_encoder_length=3072 \
  --max_decoder_length=256 \
  --num_attention_heads=16 \
  --num_hidden_layers=16 \
  --hidden_size=1024 \
  --intermediate_size=4096 \
  --block_size=64 \
  --scope=pegasus \
  --norm_type=prenorm \
  --hidden_act=relu \
  --use_bias=False \
  --rescale_embedding=True \
  --vocab_model_file=pegasus \
  --substitute_newline="<n>" \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --do_train=True \
  --do_eval=False \
  --use_tpu=True \
  --tpu_name=bigbird \
  --tpu_zone=europe-west4-a \
  --gcp_project="$GCP_PROJECT_NAME" \
  --num_tpu_cores=128 \
  --init_checkpoint=gs://bigbird-transformer/summarization/pubmed/pegasus/model.ckpt-0
