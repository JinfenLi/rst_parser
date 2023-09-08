"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""
import collections
from typing import List, Union

import numpy as np

from rst_parser.utils.rst_tree.tree import RstTree, SpanNode
from rst_parser.utils.data import sub_mainclass_dict, nuc_dict, main_subclass_dict
import nltk
from allennlp.modules.elmo import batch_to_ids

class RSTPreprocessor(object):

    def __init__(self, tokenizer, glove_dict, max_length=60):

        self.tokenizer = tokenizer
        self.glove_dict = glove_dict
        self.max_length = max_length


    def process_dis_file(self, dis_file: str):
        tree = RstTree(dis_file)
        return tree.build()


    def construct_training_features(self, item_idx, action_list, span_list, relation_list, nuclearity_list, edu_texts):
        """
        Construct training set for RST parsing.

        Args:
            item_idx: item index
            action_list: shift-reduce actions
            span_list: span list
            relation_list: relation list
            nuclearity_list: nuclearity list
            edu_texts: edu texts

        Returns:
            List[Dict[str, Any]]: a list of dict, each dict contains the following keys:
            - 'edu_input_ids': a list of edu inputs
            - 'edu_attention_mask': a list of edu attention masks
            - 'span': a list of edu spans
            - 'action': a list of shift-reduce actions, 0 for shift, 1 for reduce
            - 'nuclear': a list of nuclearity labels, 0 for SN, 1 for NN, 2 for NS
            - 'relation': a list of relation labels
        """
        features = collections.defaultdict(list)
        features['item_idx'] = item_idx
        for text in edu_texts:
            input_ids, attention_mask, glove_embs, character_ids = self.process_edus(text)
            features['edu_input_ids'].append(input_ids)
            features['edu_attention_masks'].append(attention_mask)
            features['glove_embs'].append(glove_embs)
            features['character_ids'].append(character_ids)

        features['spans'] = span_list
        features['actions'] = [-1 if a is None else (0 if a == 'Shift' else 1) for a in action_list]
        features['forms'] = [-1 if n is None else (0 if n == 'SatelliteNucleus' else (1 if n == 'NucleusNucleus' else 2)) for n in nuclearity_list]
        features['relations'] = [sub_mainclass_dict[r.lower()][1] if r is not None else -1 for r in relation_list]
        # the relation is within the range of -1-17
        assert all([(r >= -1 and r <= 17) for r in features['relations']])

        return features


    def process_edus(self, text: str):
        encode_dict = self.tokenizer(text, add_special_tokens=True)
        input_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_masks = attention_masks[: self.max_length]
        else:
            pad_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            attention_masks = attention_masks + [0] * pad_length

        tokens = nltk.word_tokenize(text)[: self.max_length]
        glove_embs = []

        for token in tokens:
            glove_embs.append(self.glove_dict.get(token, np.random.normal(-1, 1, size=300)).tolist())
        while len(glove_embs) < self.max_length:
            glove_embs.append([float(0)] * 300)
        # glove_embs = torch.tensor(glove_embs).float()

        if len(tokens) < self.max_length:
            pad_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token] * pad_length
        # elmo embedding
        character_ids = batch_to_ids([tokens])[0].tolist()



        return input_ids, attention_masks, glove_embs, character_ids



class RSTPostprocessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dis_file = ''

    def encode_tree(self, input_ids, predictions):

        edu_texts = []
        for i in range(len(input_ids)):
            edu = self.tokenizer.decode(input_ids[i, :], skip_special_tokens=True)
            edu_texts.append(edu)
        # store nodes
        cur_eid = 1
        stack = []
        while predictions:
            prediction = predictions[0]
            if len(stack) > 1 and prediction[0] == stack[-2].edu_span[0] and prediction[1] == stack[-1].edu_span[1]:
                predictions.pop(0)
                right_node = stack.pop()
                left_node = stack.pop()
                nuc_label = nuc_dict[prediction[2]]
                relation_label = list(main_subclass_dict.keys())[prediction[3]]
                if nuc_label == 'SN':
                    left_node.own_nuc = 'Satellite'
                    right_node.own_nuc = 'Nucleus'
                    left_node.own_rel = relation_label
                    right_node.own_rel = 'span'
                elif nuc_label == 'NN':
                    left_node.own_nuc = 'Nucleus'
                    right_node.own_nuc = 'Nucleus'
                    left_node.own_rel = relation_label
                    right_node.own_rel = relation_label
                elif nuc_label == 'NS':
                    left_node.own_nuc = 'Nucleus'
                    right_node.own_nuc = 'Satellite'
                    left_node.own_rel = 'span'
                    right_node.own_rel = relation_label
                new_node = SpanNode("")
                new_node.edu_span = (left_node.edu_span[0], right_node.edu_span[1])
                new_node.nuc_label = nuc_label
                new_node.rel_label = relation_label
                new_node.lnode = left_node
                new_node.rnode = right_node
                new_node.lnode.pnode = new_node
                new_node.rnode.pnode = new_node
                stack.append(new_node)
            else:
                node = SpanNode("")
                node.create_leaf_node(text=edu_texts[cur_eid - 1], edu_id=cur_eid)
                stack.append(node)
                cur_eid += 1
        stack[-1].own_nuc = 'Root'
        return stack[-1]

    def decode_tree(self, tree):
        """
        Generate dis file from RST tree.

        Args:
            tree (SpanNode): the root node of RST tree

        Returns:
            str: the content of dis file
        """


        if tree == None:
            return

        if tree.own_nuc == 'Root':
            tree.depth = 0
        else:
            tree.depth = tree.pnode.depth + 1

        cur_line = "  " * tree.depth
        if tree.edu_span[0] == tree.edu_span[1]:
            cur_line += f"( {tree.own_nuc} (leaf {tree.edu_span[0]})"
        else:
            cur_line += f"( {tree.own_nuc} (span {tree.edu_span[0]} {tree.edu_span[1]})"
        if tree.own_rel:
            cur_line += f" (rel2par {tree.own_rel})"
        if tree.text:
            cur_line += f" (text _!{tree.text}!_) )"

        cur_line += "\n"
        self.dis_file += cur_line
        if tree.lnode:
            self.decode_tree(tree.lnode)
        if tree.rnode:
            self.decode_tree(tree.rnode)

        if tree.lnode and tree.rnode:
            cur_line = "  " * tree.depth +")\n"
            self.dis_file += cur_line
        return
