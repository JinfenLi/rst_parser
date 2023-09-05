"""
    author: Jinfen Li
    GitHub: https://github.com/LiJinfen
"""
import sys
# from utils.document import Doc
# from utils.span import SpanNode

import sys
from typing import List

from rst_parser.utils.rst_tree.span import SpanNode




class RstTree(object):

    def __init__(self, dis_str: str):
        self.dis_str = dis_str


    def build(self):
        """ Build BINARY RST tree
        """
        # Build RST multi-branched tree
        tree = self.parse_dis_tree(self.dis_str)
        # Binarize it
        tree = self.binarize_tree(tree)
        # build training_set and edu_set
        node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts = [], [], [], [], [], []
        self.postorder_DFT(tree, node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts)
        return node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts


    def process_text(self, tokens: List[str]):
        """ Preprocessing token list for filtering '(' and ')' in text
        :type tokens: str
        :param tokens: list of tokens
        """
        identifier = '_!'
        within_text = False
        for (idx, tok) in enumerate(tokens):
            if identifier in tok:
                for _ in range(tok.count(identifier)):
                    within_text = not within_text
            if ('(' in tok) and within_text:
                tok = tok.replace('(', '-LB-')
            if (')' in tok) and within_text:
                tok = tok.replace(')', '-RB-')
            tokens[idx] = tok
        return tokens


    def parse_dis_tree(self, dis_str: str):
        """
        Args:
            dis_str: the .dis file content
        Returns: stack: [SpanNode: {'action': '', 'text':, '', 'token_span': [], 'text': '', 'relation': '', 'nuclearity': ''}]

        """

        tokens = dis_str.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')',
                                                                                                       ' ) ').split()

        queue = self.process_text(tokens)

        stack = []
        # multi_branches = []
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()  # Reverse to the original order
                # Parse according to the first content word
                if len(content) < 2:
                    raise ValueError("content = {}".format(content))
                own_nuc = content.pop(0)
                if own_nuc in ['Root', 'Nucleus', 'Satellite']:
                    node = SpanNode(own_nuc=own_nuc)
                    node.create_node(content)
                    stack.append(node)

                elif own_nuc == 'span':
                    # Merge
                    beginindex = int(content.pop(0))
                    endindex = int(content.pop(0))
                    stack.append(('span', beginindex, endindex))

                elif own_nuc == 'leaf':
                    # Merge
                    eduindex = int(content.pop(0))
                    stack.append(('leaf', eduindex, eduindex))
                elif own_nuc == 'rel2par':
                    # Merge
                    relation = content.pop(0)
                    stack.append(('relation', relation))
                elif own_nuc == 'text':
                    stack.append(('text', ' '.join([t.replace("_!", "") for t in content])))
                else:
                    raise ValueError(
                        "Unrecognized parsing label: {} \n\twith content = {}\n\tstack={}\n\tqueue={}".format(own_nuc,
                                                                                                              content,
                                                                                                              stack,
                                                                                                              queue))
            else:
                # else, keep push into the stack
                stack.append(token)
        # print(stack)
        return stack[-1]


    @staticmethod
    def binarize_tree(tree):
        """ Convert a general RST tree to a binary RST tree
        :type tree: instance of SpanNode
        :param tree: a general RST tree
        """
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.node_list
            # Construct binary tree
            if len(node.node_list) == 2:
                node.lnode = node.node_list[0]
                node.rnode = node.node_list[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node

            elif len(node.node_list) > 2:

                node.lnode = node.node_list.pop(0)
                newnode = SpanNode(node.node_list[0].own_nuc)
                newnode.edu_span = (node.lnode.edu_span[1] + 1, node.edu_span[1])
                newnode.own_rel = node.lnode.own_rel
                newnode.node_list += node.node_list
                # Right-branching
                node.rnode = newnode
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node

                queue.insert(0, newnode)
            # Clear node_list for the current node
            node.node_list = []
        return tree


    def postorder_DFT(self, tree, node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts):
        """ Post order traversal on binary RST tree
        :type tree: SpanNode instance
        :param tree: an binary RST tree
        :type node_list: list
        :param node_list: list of node in post order
        """
        if tree.lnode is not None:
            self.postorder_DFT(tree.lnode, node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts)
        if tree.rnode is not None:
            self.postorder_DFT(tree.rnode, node_list, action_list, span_list, relation_list, nuclearity_list, edu_texts)
        if tree.lnode is not None and tree.rnode is not None:
            sat_node = tree.lnode if tree.lnode.own_nuc == 'Satellite' else tree.rnode
            tree.nuc_label = tree.lnode.own_nuc + tree.rnode.own_nuc
            tree.rel_label = sat_node.own_rel
        node_list.append(tree)
        action_list.append("Shift" if tree.edu_span[0] == tree.edu_span[1] else "Reduce")
        span_list.append(tree.edu_span)
        relation_list.append(tree.rel_label)
        nuclearity_list.append(tree.nuc_label)
        if tree.text:
            edu_texts.append(tree.text)
        return node_list, action_list, span_list, relation_list, nuclearity_list