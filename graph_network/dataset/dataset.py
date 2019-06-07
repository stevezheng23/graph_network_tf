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
        
        self.tmp_data_path = os.path.join(self.base_path, 'tmp')
        
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
