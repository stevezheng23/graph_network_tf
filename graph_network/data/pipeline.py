import numpy as np
import tensorflow as tf

__all__ = ["GraphPipeline"]

class GraphPipeline(object):
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
        
        edge_list_tensor = [self._data_to_tensor(
            data["edge_list"], 2, max_node_size, 0, dtype=tf.int32)
            for data in dataset if "edge_list" in data]
        node_attr_tensor = [self._data_to_tensor(
            data["node_attr"], data.num_node_attr, max_node_size, 0, dtype=tf.float32)
            for data in dataset if "node_attr" in data]
        edge_attr_tensor = [self._data_to_tensor(
            data["edge_attr"], data.num_edge_attr, max_edge_size, 0, dtype=tf.float32)
            for data in dataset if "edge_attr" in data]
        graph_attr_tensor = [self._data_to_tensor(
            [data["graph_attr"]], data.num_graph_attr, 1, 0, dtype=tf.float32)
            for data in dataset if "graph_attr" in data]
        node_label_tensor = [self._data_to_tensor(
            data["node_label"], 1, max_node_size, 0, dtype=tf.int32)
            for data in dataset if "node_label" in data]
        edge_label_tensor = [self._data_to_tensor(
            data["edge_label"], 1, max_edge_size, 0, dtype=tf.int32)
            for data in dataset if "edge_label" in data]
        graph_label_tensor = [self._data_to_tensor(
            [data["graph_label"]], 1, 1, 0, dtype=tf.int32)
            for data in dataset if "graph_label" in data]
        
        edge_list_tensor = tf.stack(edge_list_tensor, axis=0) if len(edge_list_tensor) == self.data_size else None
        node_attr_tensor = tf.stack(node_attr_tensor, axis=0) if len(node_attr_tensor) == self.data_size else None
        edge_attr_tensor = tf.stack(edge_attr_tensor, axis=0) if len(edge_attr_tensor) == self.data_size else None
        graph_attr_tensor = tf.stack(graph_attr_tensor, axis=0) if len(graph_attr_tensor) == self.data_size else None
        node_label_tensor = tf.stack(node_label_tensor, axis=0) if len(node_label_tensor) == self.data_size else None
        edge_label_tensor = tf.stack(edge_label_tensor, axis=0) if len(edge_label_tensor) == self.data_size else None
        graph_label_tensor = tf.stack(graph_label_tensor, axis=0) if len(graph_label_tensor) == self.data_size else None
        
        default_int_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
        default_float_tensor = tf.constant(0.0, shape=[1,1], dtype=tf.float32)
        edge_list_dataset = (tf.data.Dataset.from_tensor_slices(edge_list_tensor) if edge_list_tensor is not None
            else tf.data.Dataset.from_tensors(default_int_tensor).repeat(self.data_size))
        node_attr_dataset = (tf.data.Dataset.from_tensor_slices(node_attr_tensor) if node_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_float_tensor).repeat(self.data_size))
        edge_attr_dataset = (tf.data.Dataset.from_tensor_slices(edge_attr_tensor) if edge_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_float_tensor).repeat(self.data_size))
        graph_attr_dataset = (tf.data.Dataset.from_tensor_slices(graph_attr_tensor) if graph_attr_tensor is not None
            else tf.data.Dataset.from_tensors(default_float_tensor).repeat(self.data_size))
        node_label_dataset = (tf.data.Dataset.from_tensor_slices(node_label_tensor) if node_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_int_tensor).repeat(self.data_size))
        edge_label_dataset = (tf.data.Dataset.from_tensor_slices(edge_label_tensor) if edge_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_int_tensor).repeat(self.data_size))
        graph_label_dataset = (tf.data.Dataset.from_tensor_slices(graph_label_tensor) if graph_label_tensor is not None
            else tf.data.Dataset.from_tensors(default_int_tensor).repeat(self.data_size))
        
        graph_dataset = tf.data.Dataset.zip((edge_list_dataset, node_attr_dataset, edge_attr_dataset,
            graph_attr_dataset, node_label_dataset, edge_label_dataset, graph_label_dataset))
        
        if self.enable_shuffle == True:
            graph_dataset = graph_dataset.shuffle(self.buffer_size, self.random_seed)

        graph_dataset = graph_dataset.batch(batch_size=self.batch_size)
        graph_dataset = graph_dataset.prefetch(buffer_size=1)

        graph_iterator = graph_dataset.make_initializable_iterator()
        graph_batch = graph_iterator.get_next()
        
        self.initializer = graph_iterator.initializer
        self.edge_list = graph_batch[0]
        self.node_attr = graph_batch[1]
        self.edge_attr = graph_batch[2]
        self.graph_attr = graph_batch[3]
        self.node_label = graph_batch[4]
        self.edge_label = graph_batch[5]
        self.graph_label = graph_batch[6]
    
    def _data_to_tensor(self,
                        data_list,
                        data_dim,
                        max_size,
                        pad,
                        dtype):
        data_list = data_list if data_list is not None else []
        data_list.extend([[pad] * data_dim] * max_size)
        return tf.convert_to_tensor(data_list[:max_size], dtype=dtype)
