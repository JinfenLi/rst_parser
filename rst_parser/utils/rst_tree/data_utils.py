"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import json
import logging
import os
import pickle
import random
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm

cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model_dependencies')

def stratified_sampling(data, num_samples):
    num_instances = len(data)
    assert num_samples < num_instances

    counter_dict = Counter(data)
    unique_vals = list(counter_dict.keys())
    val_counts = list(counter_dict.values())
    num_unique_vals = len(unique_vals)
    assert num_unique_vals > 1

    num_stratified_samples = [int(c*num_samples/num_instances) for c in val_counts]
    assert sum(num_stratified_samples) <= num_samples
    if sum(num_stratified_samples) < num_samples:
        delta = num_samples - sum(num_stratified_samples)
        delta_samples = np.random.choice(range(num_unique_vals), replace=True, size=delta)
        for val in delta_samples:
            num_stratified_samples[unique_vals.index(val)] += 1
    assert sum(num_stratified_samples) == num_samples

    sampled_indices = []
    for i, val in enumerate(unique_vals):
        candidates = np.where(data == val)[0]
        sampled_indices += list(np.random.choice(candidates, replace=False, size=num_stratified_samples[i]))
    random.shuffle(sampled_indices)

    return sampled_indices
def sample_dataset(data_path, dataset_dict, split, num_samples, seed):
    sampled_split_filename = f'{split}_split_{num_samples}_{seed}.pkl'
    if os.path.exists(os.path.join(data_path, sampled_split_filename)):
        with open(os.path.join(data_path, sampled_split_filename), 'rb') as f:
            sampled_split = pickle.load(f)
    else:
        sampled_split = stratified_sampling(dataset_dict['label'], num_samples)
        with open(os.path.join(data_path, sampled_split_filename), 'wb') as f:
            pickle.dump(sampled_split, f)

    for key in dataset_dict.keys():
        dataset_dict[key] = sampled_split if key == 'item_idx' else [dataset_dict[key][i] for i in sampled_split]

    return dataset_dict

def save_datadict(data_path, dataset_dict, split, num_samples, seed):
    for key in tqdm(dataset_dict.keys(), desc=f'Saving {split} dataset'):
        if key in dataset_dict:
            filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
            with open(os.path.join(data_path, filename), 'wb') as f:
                pickle.dump(dataset_dict[key], f)

def get_glove_dict():
    if not os.path.exists(os.path.join(cache_dir, 'glove.pkl')):
        logging.info("Downloading GloVe embeddings...")
        torch.hub.download_url_to_file("https://embs.s3.us-east-2.amazonaws.com/glove.pkl", os.path.join(cache_dir, 'glove.pkl'))
    with open(os.path.join(cache_dir, 'glove.pkl'), 'rb') as f:
        glove_dict = pickle.load(f)
    return glove_dict