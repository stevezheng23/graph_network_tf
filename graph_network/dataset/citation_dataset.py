import os.path

import numpy as np
import tensorflow as tf

from data.utility import *
from dataset.dataset import *

__all__ = ["CitationDataset"]

class CitationDataset(Dataset):
    """citation dataset"""
    def __init__(self,
                 base_path,
                 dataset_name="citation",
                 dataset_url="https://github.com/kimiyoung/planetoid/raw/master/data"):
        """initialize citation dataset"""
        super(CitationDataset, self).__init__(base_path, dataset_name, dataset_url)
    
    @property
    def remote_files(self):
        file_types = ['graph', 'test.index', 'x', 'allx', 'tx', 'y', 'ally', 'ty']
        return ['ind.{0}.{1}'.format(self.dataset_name.lower(), file_type) for file_type in file_types]
    
    @property
    def local_file(self):
        return '{0}_graph.json'.format(self.dataset_name)
    
    def _download(self):
        dataset_path = os.path.join(self.raw_data_path, self.dataset_name)
        valid_path(dataset_path)
        
        for remote_file in self.remote_files:
            data_url = '{0}/{1}'.format(self.dataset_url, remote_file)
            data_path = '{0}/{1}'.format(dataset_path, remote_file)
            download_from_url(data_url, data_path)
    
    def _process(self):
        input_path = os.path.join(self.raw_data_path, self.dataset_name)
        adjacency_list_file = '{0}/ind.{1}.graph'.format(input_path, self.dataset_name)
        adjacency_list_data = read_pickle(adjacency_list_file)
        edge_list = [[src, trg]
            for src in adjacency_list_data.keys()
            for trg in adjacency_list_data[src]]
        
        node_index_test_file = '{0}/ind.{1}.test.index'.format(input_path, self.dataset_name)
        node_index_test_data = read_text(node_index_test_file)
        node_index_test = [int(node_idx) for node_idx in node_index_test_data]
        
        node_attr_train_file = '{0}/ind.{1}.x'.format(input_path, self.dataset_name)
        node_attr_train_data = read_pickle(node_attr_train_file)
        node_attr_train = node_attr_train_data.toarray().tolist()
        train_size = len(node_attr_train)
        
        node_attr_train_dev_file = '{0}/ind.{1}.allx'.format(input_path, self.dataset_name)
        node_attr_train_dev_data = read_pickle(node_attr_train_dev_file)
        node_attr_train_dev = node_attr_train_dev_data.toarray().tolist()
        train_dev_size = len(node_attr_train_dev)
        
        node_attr_test_file = '{0}/ind.{1}.tx'.format(input_path, self.dataset_name)
        node_attr_test_data = read_pickle(node_attr_test_file)
        node_attr_test = node_attr_test_data.toarray().tolist()
        node_attr_test = zip(node_index_test, node_attr_test)
        node_attr_test = sorted(node_attr_test, key=lambda x: x[0])
        node_attr_test = [attr for _, attr in node_attr_test]
        test_size = len(node_attr_test)
        
        node_attr = node_attr_train_dev + node_attr_test
        node_size = len(node_attr)
        dev_size = int(min(test_size/2, train_dev_size-train_size))
        
        node_label_train_file = '{0}/ind.{1}.y'.format(input_path, self.dataset_name)
        node_label_train_data = read_pickle(node_label_train_file)
        node_label_train = node_label_train_data.tolist()
        
        node_label_train_dev_file = '{0}/ind.{1}.ally'.format(input_path, self.dataset_name)
        node_label_train_dev_data = read_pickle(node_label_train_dev_file)
        node_label_train_dev = node_label_train_dev_data.tolist()
        
        node_label_test_file = '{0}/ind.{1}.ty'.format(input_path, self.dataset_name)
        node_label_test_data = read_pickle(node_label_test_file)
        node_label_test = node_label_test_data.tolist()
        node_label_test = zip(node_index_test, node_label_test)
        node_label_test = sorted(node_label_test, key=lambda x: x[0])
        node_label_test = [label for _, label in node_label_test]
        
        node_label = node_label_train_dev + node_label_test
        
        train_mask = [0] * node_size
        train_mask[:train_size] = [1] * train_size
        dev_mask = [0] * node_size
        dev_mask[train_size:train_size+dev_size] = [1] * dev_size
        test_mask = [0] * node_size
        test_mask[train_dev_size:] = [1] * test_size
        
        graph_list = [{
            "edge_list": edge_list,
            "node_attr": node_attr,
            "node_label": node_label,
            "train_mask": train_mask,
            "dev_mask": dev_mask,
            "test_mask": test_mask
        }]
        
        output_path = os.path.join(self.processed_data_path, self.dataset_name)
        graph_data_file = '{0}/{1}_graph.json'.format(output_path, self.dataset_name)
        save_json(graph_list, graph_data_file)
    
    def _load(self):
        raise NotImplementedError
