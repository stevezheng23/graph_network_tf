import numpy as np
import tensorflow as tf

__all__ = ["DataPipline"]

class DataPipline(object):
    """graph pipeline"""
    def __init__(self,
                 dataset,
                 batch_size,
                 enable_shuffle,
                 buffer_size=10000,
                 random_seed=0):
        """initialize data pipeline"""
        if enable_shuffle == True:
            dataset = dataset.shuffle(buffer_size, random_seed)
        
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        
        iterator = dataset.make_initializable_iterator()
        self.batch_data = iterator.get_next()
        self.initializer = iterator.initializer
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.enable_shuffle = enable_shuffle
