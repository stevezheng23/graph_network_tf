import numpy as np
import tensorflow as tf

__all__ = ["GraphPipline"]

class GraphPipline(object):
    """graph pipeline"""
    def __init__(self,
                 dataset,
                 batch_size,
                 enable_shuffle=True,
                 random_seed=0):
        """initialize graph pipeline"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.enable_shuffle = enable_shuffle
        self.random_seed = random_seed
        
        edge_list = [data["edge_list"] for data in dataset if "edge_list" in data]
        node_attr = [data["node_attr"] for data in dataset if "node_attr" in data]
        edge_attr = [data["edge_attr"] for data in dataset if "edge_attr" in data]
        graph_attr = [data["graph_attr"] for data in dataset if "graph_attr" in data]
        node_label = [data["node_label"] for data in dataset if "node_label" in data]
        edge_label = [data["edge_label"] for data in dataset if "edge_label" in data]
        graph_label = [data["graph_label"] for data in dataset if "graph_label" in data]
