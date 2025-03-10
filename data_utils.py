import h5py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from typing import Dict, Any, Tuple, Optional


class CustomDataset(Dataset):
    def __init__(self, h5_file_path: str, channel:int,transform=None, input_size:int = 0):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.data = self.h5_file['data']
        self.time = self.h5_file['time']
        self.x = self.h5_file['x']
        self.y = self.h5_file['y']
        self.invariants = {k: v for k, v in self.h5_file['invariants'].items()}
        self.transform = transform
        
        self.n_chunks = self.data.shape[0]
        self.chunk_size = self.data.shape[-1]

        self.input_size = input_size
        self.channel = channel
    
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, index):
        """
        Get a specific chunk of data
        Returns:
            tuple: (input_data, targets)
            - input_data: tensor of shape (channels, x_dim, y_dim, chunk_size)
            - targets: dictionary of invariants for this chunk
        """
        data_chunk = self.inputs['data'][index]  # Shape: (channels, x_dim, y_dim, chunk_size)
        der_chunk = np.squeeze(np.array([[self.inputs['derived_data']['$\\Gamma_c$'][index]],
                     [self.inputs['derived_data']['$\\Gamma_n$'][index]],
                     [self.inputs['derived_data']['$\\mathcal{D}^E$'][index]],
                     [self.inputs['derived_data']['$\\mathcal{D}^U$'][index]],
                     [self.inputs['derived_data']['energy'][index]],
                     [self.inputs['derived_data']['enstrophy'][index]],
                     [self.inputs['derived_data']['time'][index]],
        ]), axis=1).transpose(1,0,2)

        x, der_x  = data_chunk[:,self.channel,...,:self.input_size], der_chunk[...,:self.input_size]
        y, der_y = data_chunk[:,self.channel,...,self.input_size:], der_chunk[...,self.input_size:]
        
        # if self.transform:
        #     inputs = self.transform(inputs)
        
        # targets = {
        #     name: data[index] for name, data in self.invariants.items()
        # }
        
        return {'x': x, 'y':y, 'der_x': der_x, 'der_y': der_y}
    
    def _load_h5_file_with_data(self, file_path:str, derived_data_key:str = "invariants"):
        """Method for loading .h5 files
        
        :returns: dict that contains name of the .h5 file as stored in the .h5 file, as well as a generator of the data
        """
        file = h5py.File(file_path)
        key = list(file.keys())[0]
        data = file[key]
        derived_data = file[derived_data_key]
        return dict(file=file, data=data, derived_data=derived_data)
    
    def load_data(self, input_file, target_files=None):
        """Loads input data and optional target data into the dataset
        Args:
            input_file (str): Name of the input .h5 file
            target_files (dict): Dictionary mapping task names to target .h5 file names
        """
        self.inputs = self._load_h5_file_with_data(input_file) #Dictionary with keys "file", "data"...
        # if target_files:
        #     for task, file_name in target_files.items():
        #         self.targets[task] = self.load_h5_file_with_data(file_name)

        # self.length = len(self.inputs['data'])
    
    def get_spatial_coords(self):
        return self.x[:], self.y[:]
    
    def get_time_for_chunk(self, index):
        start = index * self.chunk_size
        end = start + self.chunk_size
        return self.time[start:end]
    
    def close(self):
        self.h5_file.close()

class DerivedQuanities(Dataset):
    def __init__(self, h5_file_path: str, start:int, end:int, transform=None, input_size:int = 0, batch_size:int=32):
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.input_size = input_size
        self.start = start
        self.end = end

        self.index_adjustment = [self.start for i in range(batch_size)]

        self.load_data()
    
    def __len__(self):
        return self.end-self.start
    
    def __getitem__(self, index):
        """
        Get a specific chunk of data
        Returns:
            tuple: (input_data, targets)
            - input_data: tensor of shape (channels, x_dim, y_dim, chunk_size)
            - targets: dictionary of invariants for this chunk
        """
        index = list(map(lambda a, b: a + b, index, self.index_adjustment))
        x = self.inputs['data'][index]  # Shape: (channels, x_dim, y_dim,)
        y = np.squeeze(np.array([[self.inputs['derived_data']['$\\Gamma_c$'][index]],
                     [self.inputs['derived_data']['$\\Gamma_n$'][index]],
                     [self.inputs['derived_data']['$\\mathcal{D}^E$'][index]],
                     [self.inputs['derived_data']['$\\mathcal{D}^U$'][index]],
                     [self.inputs['derived_data']['energy'][index]],
                     [self.inputs['derived_data']['enstrophy'][index]],
                     [self.inputs['derived_data']['time'][index]],
        ]), axis=1).transpose(1,0,) # Shape: (7,)
        y = y[...,1].astype(np.float32)
        
        return {'x': x, 'y':y,}
    
    def _load_h5_file_with_data(self, file_path:str, data_key:str = 'data', derived_data_key:str = "invariants"):
        """Method for loading .h5 files
        
        :returns: dict that contains name of the .h5 file as stored in the .h5 file, as well as a generator of the data
        """
        file = h5py.File(file_path)
        data = file[data_key]
        self.length = data.shape[0]
        derived_data = file[derived_data_key]
        return dict(file=file, data=data, derived_data=derived_data)
    
    def load_data(self,):
        """Loads input data and optional target data into the dataset
        Args:
            input_file (str): Name of the input .h5 file
            target_files (dict): Dictionary mapping task names to target .h5 file names
        """
        self.inputs = self._load_h5_file_with_data(self.h5_file_path) #Dictionary with keys "file", "data"...
    
    def close(self):
        self.h5_file.close()


class RandomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)

        # Handle the last incomplete batch if dataset length isn't divisible by batch size
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)

def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by leveraging .h5 dataset characteristics
    Optimizes for HDF5's efficient sequential reading by using weak shuffling:
    - Batches data and shuffles batches instead of individual items
    - Reduces number of random accesses to HDF5 file
    """
    return DataLoader(
        dataset,
        batch_size=None,  # Must be disabled when using samplers
        sampler=BatchSampler(
            RandomBatchSampler(dataset, batch_size),
            batch_size=batch_size,
            drop_last=drop_last
        )
    )


def create_dataloaders(train_file: str, test_file: str, batch_size: int,input_size:int,
                              transform=None, drop_last: bool = False, channel:int = 0):
    """
    Create train and test data loaders for plasma physics data
    """
    train_dataset = CustomDataset(train_file, transform=transform, input_size=input_size, channel=channel)
    train_dataset.load_data(train_file)
    test_dataset = CustomDataset(test_file, transform=transform, input_size=input_size, channel=channel)
    test_dataset.load_data(test_file)
    
    train_loader = fast_loader(train_dataset, batch_size=batch_size, drop_last=drop_last)
    test_loader = fast_loader(test_dataset, batch_size=batch_size, drop_last=drop_last)
    
    return train_loader, test_loader