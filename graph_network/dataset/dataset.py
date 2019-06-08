import os.path

import numpy as np
import tensorflow as tf

from data.utility import *

__all__ = ["Dataset"]

class Dataset(object):
    """base dataset"""
    def __init__(self,
                 base_path,
                 dataset_name="base",
                 dataset_url=None):
        """initialize base dataset"""
        self.base_path = base_path
        valid_path(self.base_path)
        
        self.raw_data_path = os.path.join(self.base_path, 'raw')
        valid_path(self.raw_data_path)
        
        self.processed_data_path = os.path.join(self.base_path, 'processed')
        valid_path(self.processed_data_path)
        
        self.tmp_data_path = os.path.join(self.base_path, 'tmp')
        valid_path(self.tmp_data_path)
        
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
        
        self._download()
        self._process()
        self.data_list = self._load()
    
    def _download(self):
        raise NotImplementedError
    
    def _process(self):
        raise NotImplementedError
    
    def _load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,
                    idx):
        return self.data_list[idx]
