import numpy as np
import tensorflow as tf

__all__ = ["KernelDataset"]

class KernelDataset(object):
    """kernel dataset"""
    def __init__(self,
                 base_path,
                 dataset_url="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets",
                 dataset_name="kernel"):
        """initialize kernel dataset"""
        super(KernelDataset, self).__init__(base_path, dataset_url, dataset_name)
    
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
