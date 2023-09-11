"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import os
import collections
import argparse, random, logging
import numpy as np
import re
from transformers import AutoTokenizer
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from rst_parser.utils.data import sub_mainclass_dict, main_subclass_dict
def token_num_distribution():
    occur_dis = []
    relation_dis = []
    for file in os.listdir(os.path.join(args.data_dir, args.dataset, args.split)):
        if file.endswith(".dis"):
            with open(os.path.join(args.data_dir, args.dataset, args.split, file), 'r') as f:
                for line in f:
                    text = re.search(r'text _!.*_!', line)
                    if text:
                        text = text.group(0)
                        encode_dict = tokenizer(text, add_special_tokens=True)
                        input_ids = encode_dict['input_ids']
                        occur_dis.append(len(input_ids))
                    # (rel2par elaboration-object-attribute-e)
                    relation_group = re.search(r'\(rel2par [^\)]*\)', line)

                    if relation_group:
                        relation = relation_group.group(0).replace("rel2par ", "").replace(")", "").replace("(", "")
                        if relation.lower() in sub_mainclass_dict:
                            relation_dis.append(sub_mainclass_dict.get(relation.lower())[1])
                        elif relation != "span":
                            print("unrecognized relation", relation)
                        # occur_dis.append(len(text.split()))
    print("mean: ", np.mean(occur_dis))
    print("std: ", np.std(occur_dis))
    print("max: ", np.max(occur_dis))
    print("min: ", np.min(occur_dis))
    print("median: ", np.median(occur_dis))
    print("relation: ", {k: 1/v for k, v in collections.Counter(relation_dis).items()})
    weights = []
    for i, relation_label in enumerate(main_subclass_dict.keys()):
        print(relation_label)
        weights.append(1 / relation_dis.count(i))
    weights = [round(w / sum(weights), 5) for w in weights]
    print("weights: ", weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory for datasets')
    # parser.add_argument('--resource_dir', type=str, default='resources/', help='Root directory for resources')
    parser.add_argument('--dataset', type=str, default='rst_dt',
                        choices=['new_data', 'rst_dt'])
    parser.add_argument('--arch', type=str, default='bert-base-uncased',
                        help='pretrained model name')
    parser.add_argument('--split', type=str, default='TRAINING', help='Dataset split',
                        choices=['TRAINING', 'TEST'])

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.arch, strip_accents=False)
    token_num_distribution()