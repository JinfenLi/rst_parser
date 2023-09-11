# Neural-based RST Parsing And Analysis In Persuasive Discourse

This is the official PyTorch [repo](https://github.com/JinfenLi/multi_emotion_recognition) for [RST-NN-Parser](https://aclanthology.org/2021.wnut-1.30/), a deep neural network model for RST parsing.

```
Neural-based RST Parsing And Analysis In Persuasive Discourse
Jinfen Li, Lu Xiao
W-NUT 2021
```


If Multi-EmoBERT is helpful for your research, please consider citing our paper:

```
@inproceedings{li2021neural,
  title={Neural-based rst parsing and analysis in persuasive discourse},
  author={Li, Jinfen and Xiao, Lu},
  booktitle={Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)},
  pages={274--283},
  year={2021}
}
```
## Usage via Pip Package
install pip package
```
pip install rst-parser

```
use pip package
```
from rst_parser import rst_parser
tree_results, dis_results = rst_parser.parse(["The scientific community is making significant progress in understanding climate change. Researchers have collected vast amounts of data on temperature fluctuations, greenhouse gas emissions, and sea-level rise. This data shows a clear pattern of increasing global temperatures over the past century. However, there are still debates about the causes and consequences of climate change."])
```

tree_results explanation
```
each tree in the tree_results is the binary tree with the following node attributes
    text # text of an EDU
    own_rel # own relation
    edu_span # edu span
    own_nuc # own nuclearity
    lnode # left child nodes
    rnode # right child nodes
    pnode # parent node
    nuc_label # nuc_label: NN, NS, SN
    rel_label # one of the 18 relations
```


dis_results explanation
```
( Root (span 1 4)
  ( Nucleus (span 1 2) (rel2par span)
    ( Nucleus (leaf 1) (rel2par span) (text _!The scientific community is making significant progress!_) )
    ( Satellite (leaf 2) (rel2par Elaboration) (text _!in understanding climate change .!_) )
  )
  ( Satellite (span 3 4) (rel2par Elaboration)
    ( Nucleus (leaf 3) (rel2par span) (text _!Researchers have collected vast amounts of data on temperature fluctuations , greenhouse gas emissions , and sea-level rise .!_) )
    ( Satellite (leaf 4) (rel2par Contrast) (text _!This data shows a clear pattern of increasing global temperatures over the past century . However , there are still debates about the causes and consequences of climate change .!_) )
  )
)
```

## Usage via [Source Code](https://github.com/JinfenLi/rst_parser)


### Environment
create a virtual environment 
```
conda create -n emo_env python=3.9.16
```
install packages via conda first and then via pip
```
pip install -r requirements.txt
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

```
rename .env.example as .env and change the variable values in the file

### Neptune
Before running the code, you need to complete the following steps:
1. Create a [Neptune](https://neptune.ai/) account and project.
2. Edit the [NEPTUNE_API_TOKEN] and [NEPTUNE_NAME] fields in the .env file.


### Multirun
Do grid search over different configs.
```
python main.py -m \
    dataset=rst_dt \
    seed=0,1,2,3,4,5 \
```

### Evaluate checkpoint
This command evaluates a checkpoint on the train, dev, and test sets.
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
    training.eval_splits=train,dev,test \
```

### Finetune checkpoint
```
python main.py \
    training=evaluate \
    training.ckpt_path=/path/to/ckpt \
```

### Offline Mode
In offline mode, results are not logged to Neptune.
```
python main.py logger=neptune logger.offline=True
```

### Debug Mode
In debug mode, results are not logged to Neptune, and we only train/evaluate for limited number of batches and/or epochs.
```
python main.py debug=True
```

### Hydra Working Directory

Hydra will change the working directory to the path specified in `configs/hydra/default.yaml`. Therefore, if you save a file to the path `'./file.txt'`, it will actually save the file to somewhere like `logs/runs/xxxx/file.txt`. This is helpful when you want to version control your saved files, but not if you want to save to a global directory. There are two methods to get the "actual" working directory:

1. Use `hydra.utils.get_original_cwd` function call
2. Use `cfg.work_dir`. To use this in the config, can do something like `"${data_dir}/${.dataset}/${model.arch}/"`


### Config Key

- `work_dir` current working directory (where `src/` is)

- `data_dir` where data folder is

- `log_dir` where log folder is (runs & multirun)

- `root_dir` where the saved ckpt & hydra config are


---


## Example Commands

Here, we assume the following:
- The `data_dir` is `data`, which means `data_dir=${work_dir}/../data`.
- The dataset is [`RST Discourse Treebank`](https://catalog.ldc.upenn.edu/LDC2002T07).

### 1. Build dataset
The commands below are used to build pre-processed datasets, saved as pickle files. The model architecture is specified so that we can use the correct tokenizer for pre-processing.
Remember to put a xxx.yaml file in the configs/dataset folder for the dataset you want to build. 
```
python scripts/build_dataset.py --data_dir data \
    --dataset rst_dt --arch bert-base-uncased --max_length 20 --split train

python scripts/build_dataset.py --data_dir data \
    --dataset rst_dt --arch bert-base-uncased --max_length 20 --split dev

python scripts/build_dataset.py --data_dir data \
    --dataset rst_dt --arch bert-base-uncased --max_length 20 --split test

```

If the dataset is very large, you have the option to subsample part of the dataset for smaller-scale experiements. For example, in the command below, we build a train set with only 1000 train examples (sampled with seed 0).
```
python scripts/build_dataset.py --data_dir data \
    --dataset rst_dt --arch bert-base-uncased --max_length 20 --split train \
    --num_samples 10 --seed 0
```

### 2. Train RST Parser

The command below is the most basic way to run `main.py`

Noted: train_batch_size, eval_batch_size are the number of documents in a batch, mini_batch_size is the number of EDUs in a document
```
python main.py -m \
    data=rst_dt \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=1 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```

### 3. Train Model with Different Configs

**Change blstm_hidden_size**
```
python main.py -m \
    data=rst_dt \
    model.blstm_hidden_size=300 \
    model.optimizer.lr=2e-5 \
    setup.train_batch_size=1 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```
**Use Glove Embedding**
```
python main.py -m \
    data=rst_dt \
    model.blstm_hidden_size=300 \
    model.optimizer.lr=2e-5 \
    model.use_glove=True \
    setup.train_batch_size=1 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```
**Use Elmo Large Embedding**
```
python main.py -m \
    data=rst_dt \
    model.blstm_hidden_size=300 \
    model.optimizer.lr=2e-5 \
    model.edu_encoder_arch=elmo \
    model.embedding.model_size=large \
    setup.train_batch_size=1 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```
**Use BERT Embedding**
```
python main.py -m \
    data=rst_dt \
    model.blstm_hidden_size=300 \
    model.optimizer.lr=2e-5 \
    model.edu_encoder_arch=bert \
    setup.train_batch_size=1 \
    setup.accumulate_grad_batches=1 \
    setup.eff_train_batch_size=1 \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```

### 4. Evaluate Model
exp_id is the folder name under your save_dir (e.g., "rst_dt_xxx"), ckpt_path is the checkpoint under the checkpoints folder in the exp_id folder.
The results will be saved in the model_outputs folder in the exp_id folder.
```
python main.py -m \
    data=rst_dt \
    training=evaluate \
    ckpt_path = xxx \
    exp_id = xxx \
    setup.eval_batch_size=1 \
    setup.mini_batch_size=10 \
    setup.num_workers=3 \
    seed=0,1,2
```
