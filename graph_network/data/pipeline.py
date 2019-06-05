import numpy as np
import tensorflow as tf

__all__ = ["GraphPipline"]

class GraphPipline(object):
    """graph pipeline"""
    def __init__(self,
                 dataset,
                 batch_size,
                 enable_shuffle=True,
                 random_seed=0):
        """initialize graph pipeline"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.enable_shuffle = enable_shuffle
        self.random_seed = random_seed
