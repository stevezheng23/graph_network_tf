import os.path

import numpy as np
import tensorflow as tf

__all__ = ["Dataset"]

class Dataset(object):
    """base dataset"""
    def __init__(self,
                 base_path,
                 dataset_url=None,
                 dataset_name="base"):
        """initialize base dataset"""
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        
        self.raw_data_path = os.path.join(self.base_path, 'raw')
        if not os.path.exists(self.raw_data_path):
            os.mkdir(self.raw_data_path)
        
        self.processed_data_path = os.path.join(self.base_path, 'processed')
        if not os.path.exists(self.processed_data_path):
            os.mkdir(self.processed_data_path)
        
        self.dataset_url = dataset_url
        self.dataset_name = dataset_name
        self._download(self.dataset_url, self.raw_data_path)
        self._process(self.raw_data_path, self.processed_data_path)
        self.data_list = self._load(self.processed_data_path)
    
    def _download(self,
                  url,
                  path):
        raise NotImplementedError
    
    def _process(self,
                 input_path,
                 output_path):
        raise NotImplementedError
    
    def _load(self,
              path):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,
                    idx):
        return self.data_list[idx]
