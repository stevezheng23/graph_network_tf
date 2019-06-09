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
    
    def __contains__(self,
                     key):
        return key in self.keys
    
    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]
    
    @property
    def values(self):
        return [self[key] for key in self.__dict__.keys() if self[key] is not None]
    
    @property
    def num_nodes(self):
        for node in [self.node_attr, self.node_label]:
            if node is not None:
                return tf.shape(node)[0]
        
        return None
    
    @property
    def num_edges(self):
        for edge in [self.edge_list, self.edge_attr, self.edge_label]:
            if edge is not None:
                return tf.shape(edge)[0]
        
        return None
    
    @property
    def num_node_attr(self):
        if self.node_attr is not None:
            return tf.shape(self.node_attr)[-1]
        
        return None
    
    @property
    def num_edge_attr(self):
        if self.edge_attr is not None:
            return tf.shape(self.edge_attr)[-1]
        
        return None
    
    @property
    def num_graph_attr(self):
        if self.graph_attr is not None:
            return tf.shape(self.graph_attr)[-1]
        
        return None
