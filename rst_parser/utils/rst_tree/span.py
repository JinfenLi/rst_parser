#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:49

class SpanNode(object):
    """ RST tree node
    """

    def __init__(self, own_nuc):
        """ Initialization of SpanNode

        :type prop: string or None
        :param prop: property of this span
        """
        # Text of this span / Discourse relation
        self.text, self.own_rel = None, None
        # self.assignRelation = None
        # self.puretext = None
        # self.dependency = None
        # EDU span / Nucleus span (begin, end) index
        self.edu_span = None
        # self.nuc_span = None
        # # Nucleus single EDU
        # self.nuc_edu = None
        # Property
        self.own_nuc = own_nuc
        # Children node
        # Each of them is a node instance
        # N-S form (for binary RST tree only)
        self.lnode, self.rnode = None, None
        # Parent node
        self.pnode = None
        # Node list (for general RST tree only)
        self.node_list = []
        # Relation nuc_label: NN, NS, SN
        self.nuc_label = None
        self.rel_label = None
        # Relation between its left child and right child
        # self.child_relation = None
        # # Depth of this node on RST tree
        self.depth = 0
        # # Max depth of its subtree
        # self.max_depth = -1
        # # Height of this node on RST tree
        # self.height = 0
        # # level of this node, 0 for inner-sentence, 1 for inter-sentence but inner paragraph, 2 for inter-paragraph
        # self.level = 0
        # self.visited = False

    def create_node(self, content):
        """ Assign value to an SpanNode instance

        :type content: list
        :param content: content from stack
        """
        for c in content:
            if isinstance(c, SpanNode):
                # Sub-node
                self.node_list.append(c)
                c.pnode = self
            elif c[0] == 'span':
                self.edu_span = (c[1], c[2])
            elif c[0] == 'relation':
                self.own_rel = c[1]
            elif c[0] == 'leaf':
                self.edu_span = (c[1], c[1])
                # self.nuc_span = (c[1], c[1])
                # self.nuc_edu = c[1]
            elif c[0] == 'text':
                self.text = c[1]
            else:
                raise ValueError("Unrecognized property: {}".format(c[0]))

    def create_leaf_node(self, text, edu_id):
        self.text = text
        self.edu_span = (edu_id, edu_id)

    # def assign_relation(self, relation):
    #     if self.form == 'N~':
    #         # print('relation of N~ is: ', relation)
    #         self.lnode.relation = relation
    #         self.rnode.relation = relation
    #     elif self.form == 'NN':
    #         self.lnode.relation = relation
    #         self.rnode.relation = relation
    #     elif self.form == 'NS':
    #         self.lnode.relation = "span"
    #         self.rnode.relation = relation
    #     elif self.form == 'SN':
    #         self.lnode.relation = relation
    #         self.rnode.relation = "span"
    #     else:
    #         raise ValueError("Error when assign relation to node with form: {}".format(self.form))
