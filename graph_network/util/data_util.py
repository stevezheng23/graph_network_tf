import collections

import numpy as np
import tensorflow as tf

__all__ = ["Graph"]

class Graph(collections.namedtuple("Graph",
    ("edge_list", "node_attr", "edge_attr", "graph_attr",
     "node_label", "edge_label", "graph_label"))):
    pass
