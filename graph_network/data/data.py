import numpy as np
import tensorflow as tf

__all__ = ["GraphData"]

class GraphData(object):
    """graph data"""
    def __init__(self,
                 edge_list,
                 node_attr,
                 edge_attr,
                 graph_attr,
                 node_label=None,
                 edge_label=None,
                 graph_label=None):
        """initialize graph data"""
        self.edge_list = edge_list
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.graph_attr = graph_attr
        self.node_label = node_label
        self.edge_label = edge_label
        self.graph_label = graph_label
    
    def __getitem__(self,
                    key):
        return getattr(self, key)

    def __setitem__(self,
                    key,
                    value):
        setattr(self, key, value)
    
    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]
    
    def __contains__(self,
                     key):
        return key in self.keys
