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
        file_types = ['graph_indicator', 'A',
            'node_attributes', 'edge_attributes', 'graph_attributes',
            'node_labels', 'edge_labels', 'graph_labels']
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
        input_path = os.path.join(self.raw_data_path, self.dataset_name)
        node_mask_file = '{0}/{1}_graph_indicator.txt'.format(input_path, self.dataset_name)
        node_mask_data = read_text(node_mask_file)
        node_mask = [(idx, int(graph_idx)) for idx, graph_idx in enumerate(node_mask_data)]
        graph_node = { key: [idx for idx, _ in list(group)] for key, group in groupby(node_mask, lambda x: x[1]) }
        print(len(graph_node))
        edge_list_file = '{0}/{1}_A.txt'.format(input_path, self.dataset_name)
        edge_list_data = read_text(edge_list_file)
        edge_list = [[int(n.strip()) for n in edge.split(',')] for edge in edge_list_data]
        edge_list = [(idx, edge, node_mask[edge[0]-1][1]) for idx, edge in enumerate(edge_list)]
        graph_edge = { key: [(idx, edge) for idx, edge, _ in list(group)] for key, group in groupby(edge_list, lambda x: x[2]) }
        print(len(graph_edge))
        node_attr_file = '{0}/{1}_node_attributes.txt'.format(input_path, self.dataset_name)
        node_attr_data = read_text(node_attr_file) if os.path.exists(node_attr_file) else None
        if node_attr_data is not None:
            node_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in node_attr_data]
        
        edge_attr_file = '{0}/{1}_edge_attributes.txt'.format(input_path, self.dataset_name)
        edge_attr_data = read_text(edge_attr_file) if os.path.exists(edge_attr_file) else None
        if edge_attr_data is not None:
            edge_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in edge_attr_data]
        
        graph_attr_file = '{0}/{1}_graph_attributes.txt'.format(input_path, self.dataset_name)
        graph_attr_data = read_text(graph_attr_file) if os.path.exists(graph_attr_file) else None
        if graph_attr_data is not None:
            graph_attr_data = [[float(v.strip()) for v in attr.split(',')] for attr in graph_attr_data]
        
        node_label_file = '{0}/{1}_node_labels.txt'.format(input_path, self.dataset_name)
        node_label_data = read_text(node_label_file) if os.path.exists(node_label_file) else None
        if node_label_data is not None:
            node_label_data = [int(label) for label in node_label_data]
        
        edge_label_file = '{0}/{1}_edge_labels.txt'.format(input_path, self.dataset_name)
        edge_label_data = read_text(edge_label_file) if os.path.exists(edge_label_file) else None
        if edge_label_data is not None:
            edge_label_data = [int(label) for label in edge_label_data]
        
        graph_label_file = '{0}/{1}_graph_labels.txt'.format(input_path, self.dataset_name)
        graph_label_data = read_text(graph_label_file) if os.path.exists(graph_label_file) else None
        if graph_label_data is not None:
            graph_label_data = [int(label) for label in graph_label_data]
        
        graph_id_list = sorted(list(set(graph_node.keys()) & set(graph_edge.keys())))
        graph_list = [{
            "edge_list": [edge for _, edge in graph_edge[graph_id]],
            "node_attr": [node_attr_data[node_idx] for node_idx in graph_node[graph_id]] if node_attr_data is not None else None,
            "edge_attr": [edge_attr_data[edge_idx] for edge_idx, _ in graph_edge[graph_id]] if edge_attr_data is not None else None,
            "graph_attr": graph_attr_data[graph_id] if graph_attr_data is not None else None,
            "node_label": [node_label_data[node_idx] for node_idx in graph_node[graph_id]] if node_label_data is not None else None,
            "edge_label": [edge_label_data[edge_idx] for edge_idx, _ in graph_edge[graph_id]] if edge_label_data is not None else None,
            "graph_label": graph_label_data[graph_id-1] if graph_label_data is not None else None,
        } for graph_id in graph_id_list]
        
        output_path = os.path.join(self.processed_data_path, self.dataset_name)
        graph_data_file = '{0}/{1}_graph_data.json'.format(output_path, self.dataset_name)
        save_json(graph_list, graph_data_file)
    
    def _load(self):
        raise NotImplementedError
