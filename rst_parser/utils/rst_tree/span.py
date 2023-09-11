"""
    :author: Jinfen Li
    :url: https://github.com/JinfenLi
"""

class SpanNode(object):
    """ RST tree node
    """

    def __init__(self, own_nuc):
        """ Initialization of SpanNode
        :param own_nuc: nuclearity of this span
        """

        self.text, self.own_rel = None, None
        self.edu_span = None
        self.own_nuc = own_nuc
        self.lnode, self.rnode = None, None
        self.pnode = None # Parent node
        self.node_list = []
        self.nuc_label = None # nuc_label: NN, NS, SN
        self.rel_label = None

    def create_node(self, content):
        """ Assign value to an SpanNode instance

        :type content: list
        :param content: content from stack
        """
        for c in content:
            if isinstance(c, SpanNode):
                self.node_list.append(c)
                c.pnode = self
            elif c[0] == 'span':
                self.edu_span = (c[1], c[2])
            elif c[0] == 'relation':
                self.own_rel = c[1]
            elif c[0] == 'leaf':
                self.edu_span = (c[1], c[1])
            elif c[0] == 'text':
                self.text = c[1]
            else:
                raise ValueError("Unrecognized property: {}".format(c[0]))

    def create_leaf_node(self, text, edu_id):
        self.text = text
        self.edu_span = (edu_id, edu_id)
