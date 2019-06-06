import numpy as np
import tensorflow as tf

__all__ = ["CitationDataset"]

class CitationDataset(object):
    """citation dataset"""
    def __init__(self,
                 base_path,
                 dataset_url="https://github.com/kimiyoung/planetoid/raw/master/data",
                 dataset_name="citation"):
        """initialize citation dataset"""
        super(CitationDataset, self).__init__(base_path, dataset_url, dataset_name)
    
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
