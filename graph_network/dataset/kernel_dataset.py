import os.path
import shutil

import numpy as np
import tensorflow as tf

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
        shutil.rmtree(self.raw_data_path)
        os.rename(self.tmp_data_path, self.raw_data_path)
    
    def _process(self):
        raise NotImplementedError
    
    def _load(self):
        raise NotImplementedError
