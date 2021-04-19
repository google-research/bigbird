# Big Bird: Transformers for Longer Sequences

Not an official Google product.

# What is BigBird?
BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. Moreover, BigBird comes along with a theoretical understanding of the capabilities of a complete transformer that the sparse model can handle.

As a consequence of the capability to handle longer context,
BigBird drastically improves performance on various NLP tasks such as question answering and summarization.

More details and comparisons can be found in our [presentation](https://docs.google.com/presentation/d/1FdMNqG2b8XYc89_v7-_2sba7Iz6YAlXXWuMxUbrKFK0/preview).


# Citation
If you find this useful, please cite our [NeurIPS 2020 paper](https://papers.nips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html):
```
@article{zaheer2020bigbird,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```


# Code

The most important directory is `core`.
There are three main files in `core`.

*   [attention.py](bigbird/core/attention.py):
    Contains BigBird linear attention mechanism
*   [encoder.py](bigbird/core/encoder.py):
    Contains the main long sequence encoder stack
*   [modeling.py](bigbird/core/modeling):
    Contains packaged BERT and seq2seq transformer models with BigBird attention


### Colab/IPython Notebook

A quick fine-tuning demonstration for text classification is provided in
[imdb.ipynb](bigbird/classifier/imdb.ipynb)


### Create GCP Instance
Please create a project first and create an instance in a zone which has quota as follows

```bash
gcloud compute instances create \
  bigbird \
  --zone=europe-west4-a \
  --machine-type=n1-standard-16 \
  --boot-disk-size=50GB \
  --image-project=ml-images \
  --image-family=tf-2-3-1 \
  --maintenance-policy TERMINATE \
  --restart-on-failure \
  --scopes=cloud-platform

gcloud compute tpus create \
  bigbird \
  --zone=europe-west4-a \
  --accelerator-type=v3-32 \
  --version=2.3.1

gcloud compute ssh --zone "europe-west4-a" "bigbird"

```

For illustration we used instance name `bigbird` and zone `europe-west4-a`, but feel free to change them.
More details about creating Google Cloud TPU can be found in [online documentations](https://cloud.google.com/tpu/docs/creating-deleting-tpus#setup_TPU_only).


### Instalation and checkpoints
```bash
git clone https://github.com/google-research/bigbird.git
cd bigbird
pip3 install -e .
```
You can find pretrained and fine-tuned checkpoints in our [Google Cloud Storage Bucket](https://console.cloud.google.com/storage/browser/bigbird-transformer).

Optionally, you can download them using `gsutil` as
```bash
mkdir -p bigbird/ckpt
gsutil cp -r gs://bigbird-transformer/ bigbird/ckpt/
```

The storage bucket contains:
- pretrained BERT model for base(`bigbr_base`) and large (`bigbr_large`) size. It correspond to BERT/RoBERTa-like encoder only models. Following original BERT and RoBERTa implementation they are transformers with post-normalization, i.e. layer norm is happening after the attention layer. However, following [Rothe et al](https://arxiv.org/abs/1907.12461), we can use them partially in encoder-decoder fashion by coupling the encoder and decoder parameters, as illustrated in [bigbird/summarization/roberta_base.sh](bigbird/summarization/roberta_base.sh) launch script.
- pretrained Pegasus Encoder-Decoder Transformer in large size(`bigbp_large`). Again following original implementation of Pegasus, they are transformers with pre-normalization. They have full set of separate encoder-decoder weights. Also for long document summarization datasets, we have converted Pegasus checkpoints (`model.ckpt-0`) for each dataset and also provided fine-tuned checkpoints (`model.ckpt-300000`) which works on longer documents.
- fine-tuned `tf.SavedModel` for long document summarization which can be directly be used for prediction and evaluation as illustrated in the [colab nootebook](bigbird/summarization/eval.ipynb).


### Running Classification

For quickly starting with BigBird, one can start by running the classification experiment code in `classifier` directory.
To run the code simply execute

```shell
export GCP_PROJECT_NAME=bigbird-project  # Replace by your project name
export GCP_EXP_BUCKET=gs://bigbird-transformer-training/  # Replace
sh -x bigbird/classifier/base_size.sh
```


## Using BigBird Encoder instead BERT/RoBERTa

To directly use the encoder instead of say BERT model, we can use the following
code.

```python
from bigbird.core import modeling

bigb_encoder = modeling.BertModel(...)
```

It can easily replace [BERT's](https://arxiv.org/abs/1810.04805) encoder.


Alternatively, one can also try playing with layers of BigBird encoder

```python
from bigbird.core import encoder

only_layers = encoder.EncoderStack(...)
```


## Understanding Flags & Config

All the flags and config are explained in
`core/flags.py`. Here we explain
some of the important config paramaters.

`attention_type` is used to select the type of attention we would use. Setting
it to `block_sparse` runs the BigBird attention module.

```python
flags.DEFINE_enum(
    "attention_type", "block_sparse",
    ["original_full", "simulated_sparse", "block_sparse"],
    "Selecting attention implementation. "
    "'original_full': full attention from original bert. "
    "'simulated_sparse': simulated sparse attention. "
    "'block_sparse': blocked implementation of sparse attention.")
```

`block_size` is used to define the size of blocks, whereas `num_rand_blocks` is
used to set the number of random blocks. The code currently uses window size of
3 blocks and 2 global blocks. The current code only supports static tensors.

Important points to note:
* Hidden dimension should be divisible by the number of heads.
* Currently the code only handles tensors of static shape as it is primarily designed
for TPUs which only works with statically shaped tensors.
* For sequene length less than 1024, using `original_full` is advised as there
is no benefit in using sparse BigBird attention.

## Comparisons
Recently, [Long Range Arena](https://arxiv.org/pdf/2011.04006.pdf) provided a benchmark of six tasks that require longer context, and performed experiments to benchmark all existing long range transformers. The results are shown below. BigBird model, unlike its counterparts, clearly reduces memory consumption without sacrificing performance.

<img src="https://github.com/google-research/bigbird/blob/master/comparison.png" width="50%">
