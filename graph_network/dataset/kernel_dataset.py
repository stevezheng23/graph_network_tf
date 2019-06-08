import os.path

import numpy as np
import tensorflow as tf

from itertools import groupby

from data.utility import *
from dataset.dataset import *

__all__ = ["KernelDataset"]

class KernelDataset(Dataset):
    """kernel dataset"""
    def __init__(self,
                 base_path,
                 dataset_name="kernel",
                 dataset_url="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets"):
        """initialize kernel dataset"""
        super(KernelDataset, self).__init__(base_path, dataset_name, dataset_url)
    
    @property
    def remote_files(self):
        file_types = ['A', 'node_attributes', 'edge_attributes', 'graph_indicator', 'graph_labels']
        return ['{0}_{1}.txt'.format(self.dataset_name, file_type) for file_type in file_types]
    
    @property
    def local_file(self):
        return 'data.{0}'.format(self.dataset_name)
    
    def _download(self):
        data_url = '{0}/{1}.zip'.format(self.dataset_url, self.dataset_name)
        data_path = '{0}/{1}.zip'.format(self.tmp_data_path, self.dataset_name)
        download_from_url(data_url, data_path)
        extract_zip(data_path, self.tmp_data_path)
        os.remove(data_path)
        src_path = os.path.join(self.tmp_data_path, self.dataset_name)
        dest_path = os.path.join(self.raw_data_path, self.dataset_name)
        clear_path(dest_path)
        os.rename(src_path, dest_path)
    
    def _process(self):
        dataset_path = os.path.join(self.raw_data_path, self.dataset_name)
        graph_mask_file = '{0}/{1}_graph_indicator.txt'.format(dataset_path, self.dataset_name)
        graph_mask_data = read_file(graph_mask_file)
        graph_mask_data = [int(graph_id) for graph_id in graph_mask_data]
        
        edge_list_file = '{0}/{1}_A.txt'.format(dataset_path, self.dataset_name)
        edge_list_data = read_file(edge_list_file)
        edge_list_data = [[int(n.strip()) for n in edge.split(',')] for edge in edge_list_data]
        
        node_attr_file = '{0}/{1}_node_attributes.txt'.format(dataset_path, self.dataset_name)
        node_attr_data = read_file(node_attr_file) if os.path.exists(node_attr_file) else None
        if node_attr_data is not None:
            node_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in node_attr_data]
        
        edge_attr_file = '{0}/{1}_edge_attributes.txt'.format(dataset_path, self.dataset_name)
        edge_attr_data = read_file(edge_attr_file) if os.path.exists(edge_attr_file) else None
        if edge_attr_data is not None:
            edge_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in edge_attr_data]
        
        graph_attr_file = '{0}/{1}_graph_attributes.txt'.format(dataset_path, self.dataset_name)
        graph_attr_data = read_file(graph_attr_file) if os.path.exists(graph_attr_file) else None
        if graph_attr_data is not None:
            graph_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in graph_attr_data]
        
        node_label_file = '{0}/{1}_node_labels.txt'.format(dataset_path, self.dataset_name)
        node_label_data = read_file(node_label_file) if os.path.exists(node_label_file) else None
        if node_label_data is not None:
            node_label_data = [int(label) for label in node_label_data]
        
        edge_label_file = '{0}/{1}_edge_labels.txt'.format(dataset_path, self.dataset_name)
        edge_label_data = read_file(edge_label_file) if os.path.exists(edge_label_file) else None
        if edge_label_data is not None:
            edge_label_data = [int(label) for label in edge_label_data]
        
        graph_label_file = '{0}/{1}_graph_labels.txt'.format(dataset_path, self.dataset_name)
        graph_label_data = read_file(graph_label_file) if os.path.exists(graph_label_file) else None
        if graph_label_data is not None:
            graph_label_data = [int(label) for label in graph_label_data]
    
    def _load(self):
        raise NotImplementedError
