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
        super(CitationDataset, self).__init__(base_path, dataset_url, dataset_name)
    
    @property
    def remote_files(self):
        file_types = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph', 'test.index']
        return ['ind.{0}.{1}'.format(self.dataset_name.lower(), file_type) for file_type in file_types]
    
    @property
    def local_file(self):
        return 'data.{0}'.format(self.dataset_name)
    
    def _download(self):
        for remote_file in self.remote_files:
            data_url = '{0}/{1}'.format(self.dataset_url, remote_file)
            data_path = os.path.join(self.raw_data_path, remote_file)
            download_from_url(data_url, data_path)
    
    def _process(self):
        raise NotImplementedError
    
    def _load(self):
        raise NotImplementedError
