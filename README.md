# Discogs-VINet

This repository contains the code to train and evaluate Discogs-VINet models for musical version identification (VI), also known as, cover song identification (CSI). The model and the development dataset are discussed in our [ISMIR2024 paper](https://arxiv.org/abs/2410.17400): "Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata". 

*Note:* I am keeping the repository updated with new experiments so the main branch has different features from the ISMIR submission. If you want to access the ISMIR2024 code just do `git checkout ISMIR2024`.

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

For training and evaluating models we use the [Discogs-VI-YT dataset](https://mtg.github.io/discogs-vi-dataset/).

## Train a Model

We use YAML config files to determine model and training parameters. They are located in `configs/` directory. 

*Note:* For monitoring training we use Wandb. If you don't want to use it simply pass `--no-wandb` as indicated below.

```bash
(discogs-vinet) [oaraz@hpcmtg1 Discogs-VINet]$ python train.py -h
```

```text
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

## Pre-trained Models

We provide the pre-trained Discogs-VINet's weights and corresponding configuration file in `logs/checkpoints/Discogs-VINet/`. This model corresponds to the one described in ISMIR2024.

Later, we participated in the MIREX2024 Cover Song Identification task with the improved, Discogs-VINet-MIREX model. You can find the short technical report [here](https://futuremirex.com/portal/wp-content/uploads/2024/11/R_Oguz_Araz-MIREX2024.pdf). Our submission came in 2nd place! However, since ByteCover2 is not open source, it is known to be not reproducible, and it has x8 times more parameters, we *are* the open-source and reproducible winners. 

As described in the technical report, we overfitted an improved version of Discogs-VINet to the *full* Discogs-VI-YT and we called this model Discogs-VINet-MIREX-full_set. However, I think the performance of this new model trained on only the training partition (same as Discogs-VINet) is interesting for comparison. Unfortunately I did not report this metric on the technical paper and I do not know what to call this model. I will share the model and its metrics here soon.


## Evaluate a Model

As discussed in the paper, we included almost all of the Da-TACOS benchmark set' and SHS100K2-TEST's cliques in our test set–and added more. However, if for any reason, you still want to evaluate on either dataset, our evaluation script can do with it.

```bash
(discogs-vinet) [oaraz@hpcmtg1 Discogs-VINet]$ python evaluate.py -h
```

```text
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

You can run inference on a directory of files with `infer.py`.

```bash
(discogs-vinet) [oaraz@hpcmtg1 Discogs-VINet]$ python infer.py -h
```

```text
(discogs-vinet2) [raraz@node027 Discogs-VINet]$ python inference.py -h
[   INFO   ] MusicExtractorSVM: no classifier models were configured by default
usage: inference.py [-h] [--granularity {track,chunk}] [--overlap OVERLAP] [--fp16] [--num-workers NUM_WORKERS] [--no-gpu] [--disable-amp] [--num-partitions NUM_PARTITIONS] [--partition PARTITION] input_dir config_path output_dir

positional arguments:
  input_dir             Path to the directory containing the audio files.
  config_path           Path to the configuration file of the trained model. The config will be used to find model weigths.
  output_dir            Path to the output directory. The tree structure of the input directory will be replicated here.

options:
  -h, --help            show this help message and exit
  --granularity {track,chunk}
                        Embedding granularity level to use for extraction. Track-level embeddings create a single embedding for the whole track. Chunk-level embeddings create embeddings using the model's context length. (default: track)
  --overlap OVERLAP     Overlap ratio to use during chunk-level embedding extraction. (default: None)
  --fp16                Store the embeddings with FP16 precision to save space. (default: False)
  --num-workers NUM_WORKERS
                        Number of workers to use in the DataLoader. (default: 8)
  --no-gpu              Flag to disable the GPU. If not provided, the GPU will be used if available. (default: False)
  --disable-amp         Flag to disable Automatic Mixed Precision for inference. If not provided, AMP usage will depend on the model config file. (default: False)
  --num-partitions NUM_PARTITIONS
                        Number of partitions to split the input files into. If greater than 1, the script will process only the files corresponding to the specified partition. (default: 1)
  --partition PARTITION
                        Partition number to process (default: 0)
```

## MIREX Submission

If you want to use the MIREX submission format, you can use `mirex.py` as the following.

```text
(discogs-vinet2) [raraz@node027 Discogs-VINet]$ python mirex.py -h
usage: mirex.py [-h] [--disable-amp] [--num-workers NUM_WORKERS] config_path collection_list_file query_list_file working_directory output_file

This script is intended for running inference on a set of files and computing their pairwise similarities. It creates a csv file following the MIREX guidelines provided in https://www.music-ir.org/mirex/wiki/2024:Cover_Song_Identification. Additional to the specified arguments in the guideline, we added the ability
to disable Automatic Mixed Precision (AMP) for inference. AMP is enabled by default if not specified in the model configuration file. You can also provide the number of workers to use in the DataLoader. The script loads a pre-trained model and computes the pairwise distances between the query versions and the
candidates in the collection. The output is a tab-separated file containing the pairwise distances between the query versions and the candidates. If you wish to use GPU for inference you should add the CUDA_VISIBLE_DEVICES environment variable to the command line. For example, to use GPU 0, you should run:
CUDA_VISIBLE_DEVICES=0 python main.py <collection_list_file> <query_list_file> <working_directory> <output_file> [--disable-amp] [--num-workers]

positional arguments:
  config_path           Path to the configuration file of the trained model. The config should point to the model weigths.
  collection_list_file  Text file containing <number of candidates> full path file names for the <number of candidates> audio files in the collection (including the <number of queries> query documents). Example: /path/to/coversong/collection.txt
  query_list_file       Text file containing the <number of queries> full path file names for the <number of queries> query documents.
  working_directory     Full path to a temporary directory where submission will have write access for caching features or calculations.
  output_file           Full path to file where submission should output the similarity matrix (<number of candidates> header rows + <number of queries> x <number of candidates> data matrix).

options:
  -h, --help            show this help message and exit
  --disable-amp         Flag to disable Automatic Mixed Precision for inference. If not provided, AMP usage will depend on the model config file. (default: False)
  --num-workers NUM_WORKERS
                        Number of workers to use in the DataLoader. (default: 10)
```

## Citation

Please cite the following publication when using the ISMIR2024 model or the dataset:

> R. O. Araz, X. Serra, and D. Bogdanov, "Discogs-VI: A musical version identification dataset based on public editorial metadata," in Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR), San Francisco, CA, USA, 2024.

```bibtex
@inproceedings{araz2024-discogsvi,
 title = {Discogs-{VI}: {A} musical version identification dataset based on public editorial metadata},
 author = {Araz, R. Oguz and Serra, Xavier and Bogdanov, Dmitry},
 booktitle = {Proceedings of the 25th {International} {Society} for {Music} {Information} {Retrieval} {Conference} ({ISMIR})},
 address   = {San Francisco, CA, USA},
 year = {2024},
}
```

Please cite the following when using the MIREX2024 model:

> R. O. Araz, J. Serrà, Y. Mitsufuji, X. Serra, and D. Bogdanov, “Discogs-VINet-MIREX,” in Late-Breaking and Demo Session of the 25th International Society for Music Information Retrieval Conference (ISMIR), San Francisco, CA, USA, 2024.

```bibtex
  @misc{araz2024-discogsvinetmirex,
  title        = {Discogs-{VINet}-{MIREX}},
  author       = {Araz, R. Oguz and Serrà, Joan and Mitsufuji, Yuki and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {{Late-{Breaking} and {Demo} {Session} of the 25th {International} {Society} for {Music} {Information} {Retrieval} {Conference} ({ISMIR})}},
  address   = {San Francisco, CA, USA},
  year         = {2024},
}
```