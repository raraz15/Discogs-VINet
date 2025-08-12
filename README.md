# Discogs-VINet

This repository contains the code to train and evaluate the Discogs-VINet model on the Discogs-VI dataset. It also contains the code used for CQT extraction. The dataset and the model are discussed in detail in our ISMIR 2024 paper: "Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata"

Contact: <recepoguz.araz@upf.edu>

## Installation

We use Python 3.11.8 for running the code. Please install the conda environment using:

```bash
git clone https://github.com/raraz15/Discogs-VINet
cd Discogs-VINet
conda env create -f env.yaml
conda activate discogs-vinet
```

## Dataset

For training the model we use the Discogs-VI-YT dataset. You can Download it from: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13983028.svg)](https://doi.org/10.5281/zenodo.13983028)
For more information about the dataset please [see](https://mtg.github.io/discogs-vi-dataset/).

## Train a Model

We use YAML config files to determine model and training parameters. They are located in `configs/` directory.

```bash
(discogs-vinet) oaraz@hpcmtg1 Discogs-VINet]$ python train.py -h
usage: train.py [-h] [--epochs EPOCHS] [--save-frequency SAVE_FREQUENCY] [--eval-frequency EVAL_FREQUENCY] [--similarity-search {MCSS,NNS}] [--chunk-size CHUNK_SIZE] [--pre-eval] [--num-workers NUM_WORKERS] [--no-wandb] [--wandb-id WANDB_ID] config_path

Training script for the model.

positional arguments:
  config_path           Path to the configuration file.

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to train the model. (default: 50)
  --save-frequency SAVE_FREQUENCY
                        Save the model every N epochs. (default: 1)
  --eval-frequency EVAL_FREQUENCY
                        Evaluate the model every N epochs. (default: 5)
  --similarity-search {MCSS,NNS}, -s {MCSS,NNS}
                        Similarity search function to use for the evaluation. MCSS: Maximum Cosine Similarity Search, NNS: Nearest Neighbour Search. (default: MCSS)
  --chunk-size CHUNK_SIZE, -b CHUNK_SIZE
                        Chunk size to use during metrics calculation. (default: 1024)
  --pre-eval            Evaluate the model before training. (default: False)
  --num-workers NUM_WORKERS
                        Number of workers to use in the DataLoader. (default: 10)
  --no-wandb            Do not use wandb to log experiments. (default: False)
  --wandb-id WANDB_ID   Wandb id to resume an experiment. (default: None)
```

## Evaluate a Model

We provide Discogs-VINet weights and configuration file in `logs/checkpoints/Discogs-VINet/`.

```bash
(discogs-vinet) [oaraz@hpcmtg1 Discogs-VINet]$ python evaluate.py -h
usage: evaluate.py [-h] [--output-dir OUTPUT_DIR] [--similarity-search {MCSS,NNS}] [--features-dir FEATURES_DIR] [--chunk-size CHUNK_SIZE] [--num-workers NUM_WORKERS] [--no-gpu] config_path test_cliques

positional arguments:
  config_path           Path to the configuration file of the trained model. The config will be used to find model weigths
  test_cliques          Path to the test cliques.json file. Can be SHS100K, Da-TACOS or DiscogsVI.

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Path to the output directory. (default: None)
  --similarity-search {MCSS,NNS}, -s {MCSS,NNS}
                        Similarity search function to use for the evaluation. MCSS: Maximum Cosine Similarity Search, NNS: Nearest Neighbour Search. (default: MCSS)
  --features-dir FEATURES_DIR, -f FEATURES_DIR
                        Path to the features directory. Optional, by default uses the path in the config file. (default: None)
  --chunk-size CHUNK_SIZE, -b CHUNK_SIZE
                        Chunk size to use during metrics calculation. (default: 512)
  --num-workers NUM_WORKERS
                        Number of workers to use in the DataLoader. (default: 10)
  --no-gpu              Flag to disable GPU. If not provided, the GPU will be used if available. (default: False)
```

## Inference

```bash
(discogs-vinet) [oaraz@hpcmtg1 Discogs-VINet]$ python inference.py -h
usage: inference.py [-h] [--granularity {track,chunk}] [--overlap OVERLAP] [--fp16] [--num-workers NUM_WORKERS] [--no-gpu] [--num-partitions NUM_PARTITIONS] [--partition PARTITION] input_dir config_path output_dir

positional arguments:
  input_dir             Path to the directory containing the audio files.
  config_path           Path to the configuration file of the trained model. The config will be used to find model weigths.
  output_dir            Path to the output directory. The tree structure of the input directory will be replicated here.

options:
  -h, --help            show this help message and exit
  --granularity {track,chunk}
                        Embedding granularity level to use for extraction. Track-level embeddings create a single embedding for the whole track. Chunk-level embeddings create embeddings using the model's context length. (default: None)
  --overlap OVERLAP     Overlap ratio to use during chunk-level embedding extraction. (default: None)
  --fp16                Store the embeddings with FP16 precision to save space. (default: False)
  --num-workers NUM_WORKERS
                        Number of workers to use in the DataLoader. (default: 8)
  --no-gpu              Flag to disable the GPU. If not provided, the GPU will be used if available. (default: False)
  --num-partitions NUM_PARTITIONS
  --partition PARTITION
```

## Citation

Please cite the following publication when using the model or the dataset:

> R. O. Araz, X. Serra, and D. Bogdanov, "Discogs-VI: A musical version identification dataset based on public editorial metadata," in Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR), 2024.

```bibtex
@inproceedings{araz_discogs-vi_2024,
 title = {Discogs-{VI}: {A} musical version identification dataset based on public editorial metadata},
 booktitle = {Proceedings of the 25th {International} {Society} for {Music} {Information} {Retrieval} {Conference} ({ISMIR})},
 author = {Araz, R. Oguz and Serra, Xavier and Bogdanov, Dmitry},
 year = {2024},
}
```
