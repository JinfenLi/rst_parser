"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import argparse, random, logging
import collections
import pickle
from collections import defaultdict as ddict
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)


from rst_parser.utils.data import dataset_info, data_keys
from rst_parser.utils.rst_tree.edu_segmenter import EDUSegmenter
from rst_parser.utils.rst_tree.processor import RSTPreprocessor
from rst_parser.utils.rst_tree.data_utils import save_datadict, get_glove_dict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datadict(data_path, split, num_samples, seed):
    dataset_dict = {}
    for key in tqdm(data_keys, desc=f'Loading {split} dataset'):
        if key in dataset_dict:
            filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
            with open(os.path.join(data_path, filename), 'rb') as f:
                dataset_dict[key] = pickle.load(f)
    return dataset_dict


def main():

    set_random_seed(args.seed)

    assert args.split is not None and args.arch is not None
    assert args.num_samples is None or args.num_samples >= 1
    num_examples = args.num_samples
    if args.dataset != 'new_data':

        max_length = dataset_info[args.dataset]['max_length'][args.arch]
    else:
        args.split = 'test'
        max_length = 20

    tokenizer = AutoTokenizer.from_pretrained(args.arch, strip_accents=False)
    glove_dict = get_glove_dict()
    preprocessor = RSTPreprocessor(tokenizer, glove_dict, max_length)
    data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.split)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dataset_dict = ddict(list)


    if args.dataset == 'rst_dt':
        # get all the files ending with .dis in the TRAINING and TESTING folders
        file_path = os.path.join(args.data_dir, args.dataset, 'TRAINING' if args.split != 'test' else "TEST")
        files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.dis')]
        random_idx = random.sample(range(len(files)), len(files))
        if args.split == 'dev':
            files = [files[idx] for idx in random_idx[:40]]
        elif args.split == 'train':
            files = [files[idx] for idx in random_idx[40:]]
        if num_examples is not None:
            assert num_examples <= len(files)
        if num_examples is not None:
            files = random.sample(files, num_examples)

        for idx, file in enumerate(tqdm(files, desc=f'Building {args.split} dataset')):

            with open(file, 'r') as f:
                dis_str = f.read()
                node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts = preprocessor.process_dis_file(dis_str)
                feature_dict = preprocessor.construct_training_features(idx, action_list, span_list, relation_list, nuclearity_list, edu_texts)
                for key in data_keys:
                    dataset_dict[key].append(feature_dict[key])

    elif args.dataset == 'new_data':
        with open("data/new_data/txt_files/doc.txt", 'r') as f:
            doc = f.read()
        cache_dir = os.path.join("../rst_parser", "segmenter")
        edu_segmenter = EDUSegmenter(cache_dir)
        tokenized_sentences, end_boundaries = edu_segmenter([doc])
        rst_preprocessor = RSTPreprocessor(tokenizer, max_length)
        for i, doc in enumerate([doc]):
            sentence = tokenized_sentences[i]
            boundaries = end_boundaries[i]
            cur_start = 0
            edus = []
            for b in boundaries:
                edus.append(" ".join(sentence[cur_start: b + 1]))
                cur_start = b + 1

            if len(edus) < 2:
                raise ValueError(f"Document {i} has less than 2 EDUs")
            features = collections.defaultdict(list)
            for edu in edus:
                input_ids, attention_mask = rst_preprocessor.process_edus(edu)
                features['edu_input_ids'].append(input_ids)
                features['edu_attention_masks'].append(attention_mask)
            dataset_dict['item_idx'].append(i)
            dataset_dict['edu_input_ids'].append(features['edu_input_ids'])
            dataset_dict['edu_attention_masks'].append(features['edu_attention_masks'])

    save_datadict(data_path, dataset_dict, args.split, args.num_samples, args.seed)

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory for datasets')
    # parser.add_argument('--resource_dir', type=str, default='resources/', help='Root directory for resources')
    parser.add_argument('--dataset', type=str, default='rst_dt',
                        choices=['new_data', 'rst_dt'])
    parser.add_argument('--arch', type=str, default='bert-base-uncased',
                        help='pretrained model name')
    parser.add_argument('--seg_model', type=str, default='bert_uncased', help='Segmentation model',
                        choices=['bert_uncased', 'bert_cased', 'bart'])
    parser.add_argument('--split', type=str, default='test', help='Dataset split', choices=['train', 'dev', 'test'])
    parser.add_argument('--stratified_sampling', type=bool, default=False, help='Whether to use stratified sampling')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    main()
