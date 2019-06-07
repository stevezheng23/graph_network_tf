import numpy as np
import tensorflow as tf

__all__ = ["GraphPipline"]

class GraphPipline(object):
    """graph pipeline"""
    def __init__(self,
                 dataset,
                 max_node_size,
                 max_edge_size,
                 batch_size,
                 buffer_size=10000,
                 random_seed=0,
                 enable_shuffle=True):
        """initialize graph pipeline"""
        self.dataset = dataset
        self.data_size = len(dataset)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.random_seed = random_seed
        self.enable_shuffle = enable_shuffle
        
        edge_list = [data["edge_list"] for data in dataset if "edge_list" in data]
        node_attr = [data["node_attr"] for data in dataset if "node_attr" in data]
        edge_attr = [data["edge_attr"] for data in dataset if "edge_attr" in data]
        graph_attr = [data["graph_attr"] for data in dataset if "graph_attr" in data]
        node_label = [data["node_label"] for data in dataset if "node_label" in data]
        edge_label = [data["edge_label"] for data in dataset if "edge_label" in data]
        graph_label = [data["graph_label"] for data in dataset if "graph_label" in data]
        
        default_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
        edge_list_tensor = tf.convert_to_tensor(edge_list, dtype=tf.int32) if len(edge_list) == self.data_size else None
        node_attr_tensor = tf.convert_to_tensor(node_attr, dtype=tf.int32) if len(node_attr) == self.data_size else None
        edge_attr_tensor = tf.convert_to_tensor(edge_attr, dtype=tf.int32) if len(edge_attr) == self.data_size else None
        graph_attr_tensor = tf.convert_to_tensor(graph_attr, dtype=tf.int32) if len(graph_attr) == self.data_size else None
        node_label_tensor = tf.convert_to_tensor(node_label, dtype=tf.int32) if len(node_label) == self.data_size else None
        edge_label_tensor = tf.convert_to_tensor(edge_label, dtype=tf.int32) if len(edge_label) == self.data_size else None
        graph_label_tensor = tf.convert_to_tensor(graph_label, dtype=tf.int32) if len(graph_label) == self.data_size else None
        
        edge_list_dataset = (tf.data.Dataset.from_tensor_slices(edge_list_tensor) if edge_list_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        node_attr_dataset = (tf.data.Dataset.from_tensor_slices(node_attr_tensor) if node_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        edge_attr_dataset = (tf.data.Dataset.from_tensor_slices(edge_attr_tensor) if edge_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        graph_attr_dataset = (tf.data.Dataset.from_tensor_slices(graph_attr_tensor) if graph_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        node_label_dataset = (tf.data.Dataset.from_tensor_slices(node_label_tensor) if node_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        edge_label_dataset = (tf.data.Dataset.from_tensor_slices(edge_label_tensor) if edge_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        graph_label_dataset = (tf.data.Dataset.from_tensor_slices(graph_label_tensor) if graph_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_tensor).repeat(self.data_size))
        
        graph_dataset = tf.data.Dataset.zip([edge_list_dataset, node_attr_dataset, edge_attr_dataset,
            graph_attr_dataset, node_label_dataset, edge_label_dataset, graph_label_dataset])
        
        if self.enable_shuffle == True:
            graph_dataset = graph_dataset.shuffle(self.buffer_size, self.random_seed)

        graph_dataset = graph_dataset.batch(batch_size=self.batch_size = batch_size)
        graph_dataset = graph_dataset.prefetch(buffer_size=1)

        graph_iterator = graph_dataset.make_initializable_iterator()
        graph_batch = graph_iterator.get_next()
