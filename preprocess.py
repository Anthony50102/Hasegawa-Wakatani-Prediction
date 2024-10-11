import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import time
import gc

@dataclass
class ProcessingConfig:
    chunk_size: int
    subsample_rate: int
    test_split: float
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    normalizer: Optional[object] = None
    normalizer_size: Optional[int] = None
    output_performance_report: bool = False

class HDF5Processor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.data_shape = None
        self.performance_metrics = {}

    def process_file(self, train_output_file: str, test_output_file: str, config: ProcessingConfig) -> Dict[str, Any]:
        start_time = time.time()
        
        with h5py.File(self.input_file, 'r') as f_in:
            total_timesteps = f_in['data'].shape[0]
            start_idx, end_idx = self._validate_range(config.start_idx, config.end_idx, total_timesteps)

            if config.normalizer is not None:
                self._normalize(f_in, config.normalizer, start_idx, end_idx, config.normalizer_size)
            
            subsampled_data = self._subsample_data(f_in, start_idx, end_idx, config.subsample_rate)
            metadata = self._split_and_save_data(subsampled_data, train_output_file, test_output_file, f_in, config)
            
            del subsampled_data
            gc.collect()

        self.performance_metrics['total_processing_time'] = time.time() - start_time
        
        if config.output_performance_report:
            self._output_performance_report()

        return metadata

    def _normalize(self, f_in, normalizer, start_idx, end_idx, chunk_size):
        normalize_start_time = time.time()
        
        data = f_in['data'][start_idx:end_idx]
        data = torch.from_numpy(data)
        
        if data.shape[0] % chunk_size != 0:
            print(f"Warning: Dataset size {data.shape[0]} is not divisible by chunk size {chunk_size}.")

        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i+chunk_size]
            normalizer.partial_fit(chunk, chunk_size)
            del chunk
        
        # Check normalization
        normalized_means = []
        normalized_stds = []
        original_means = []
        original_stds = []
        
        for i in range(0, data.shape[0], chunk_size):
            original_chunk = data[i:i+chunk_size]
            normalized_chunk = normalizer.forward(original_chunk)
            
            original_means.append(original_chunk.mean().item())
            original_stds.append(original_chunk.std().item())
            normalized_means.append(normalized_chunk.mean().item())
            normalized_stds.append(normalized_chunk.std().item())
            
            del original_chunk, normalized_chunk

        print(f"Original data - Mean: {np.mean(original_means):.4f}, Std: {np.mean(original_stds):.4f}")
        print(f"Normalized data - Mean: {np.mean(normalized_means):.4f}, Std: {np.mean(normalized_stds):.4f}")
        
        if not (np.isclose(np.mean(normalized_means), 0, atol=1e-2) and np.isclose(np.mean(normalized_stds), 1, atol=1e-2)):
            print("Warning: Normalization may not have worked as expected.")
        else:
            print("Normalization check passed.")

        del data, original_means, original_stds, normalized_means, normalized_stds
        gc.collect()

        self.performance_metrics['normalization_time'] = time.time() - normalize_start_time


    def _subsample_data(self, f_in: h5py.File, start_idx: int, end_idx: int, subsample_rate: int) -> Dict[str, np.ndarray]:
        subsample_start_time = time.time()
        
        data = f_in['data'][start_idx:end_idx:subsample_rate]
        time_data = f_in['time'][start_idx:end_idx:subsample_rate]
        self.data_shape = data.shape
        
        invariants = {member: f_in['invariants'][member][start_idx:end_idx:subsample_rate] for member in f_in['invariants']}
        
        self.performance_metrics['subsample_time'] = time.time() - subsample_start_time
        
        return {'data': data, 'time': time_data, 'invariants': invariants}

    def _save_split(self, subsampled_data: Dict[str, np.ndarray], output_file: str,
                    input_file: h5py.File, config: ProcessingConfig, 
                    chunk_indices: List[int], split_name: str) -> Dict[str, Any]:
        save_start_time = time.time()
        
        n_chunks = len(chunk_indices)
        _, k_channels, x_dim, y_dim = self.data_shape
        
        with h5py.File(output_file, 'w') as f_out:
            chunked_shape = (n_chunks, k_channels, x_dim, y_dim, config.chunk_size)
            chunked_data = f_out.create_dataset('data', chunked_shape, dtype=subsampled_data['data'].dtype)
            
            for new_idx, orig_idx in enumerate(chunk_indices):
                start = orig_idx * config.chunk_size
                end = start + config.chunk_size
                data_chunk = subsampled_data['data'][start:end]
                data_chunk = config.normalizer.forward(data_chunk)
                chunked_data[new_idx] = np.transpose(data_chunk, (1, 2, 3, 0))
                del data_chunk
            
            time_data = np.concatenate([subsampled_data['time'][i*config.chunk_size:(i+1)*config.chunk_size] 
                                       for i in chunk_indices])
            f_out.create_dataset("time", data=time_data)
            
            f_out.create_dataset("x", data=input_file["x"][:])
            f_out.create_dataset("y", data=input_file["y"][:])
            
            invariants_group = f_out.create_group("invariants")
            for member, invariant_data in subsampled_data['invariants'].items():
                chunked_invariants = invariants_group.create_dataset(
                    member,
                    (n_chunks, config.chunk_size),
                    dtype=invariant_data.dtype
                )
                
                for new_idx, orig_idx in enumerate(chunk_indices):
                    start = orig_idx * config.chunk_size
                    end = start + config.chunk_size
                    chunked_invariants[new_idx] = invariant_data[start:end]
        
        self.performance_metrics[f'{split_name}_save_time'] = time.time() - save_start_time
        
        return {
            'n_chunks': n_chunks,
            'chunk_size': config.chunk_size,
            'processed_shape': chunked_shape,
            'actual_timesteps': n_chunks * config.chunk_size,
            'split': split_name,
            'invariants': list(subsampled_data['invariants'].keys())
        }

    def _split_and_save_data(self, subsampled_data: Dict[str, np.ndarray], 
                            train_output_file: str, test_output_file: str,
                            input_file: h5py.File, config: ProcessingConfig) -> Dict[str, Any]:
        split_start_time = time.time()
        
        n_timesteps = subsampled_data['data'].shape[0]
        n_chunks = n_timesteps // config.chunk_size
        actual_timesteps = n_chunks * config.chunk_size
        
        for key in subsampled_data:
            if key == 'invariants':
                for member in subsampled_data[key]:
                    subsampled_data[key][member] = subsampled_data[key][member][:actual_timesteps]
            else:
                subsampled_data[key] = subsampled_data[key][:actual_timesteps]
        
        n_test_chunks = int(n_chunks * config.test_split)
        n_train_chunks = n_chunks - n_test_chunks
        
        all_chunk_indices = np.arange(n_chunks)
        np.random.shuffle(all_chunk_indices)
        test_chunk_indices = all_chunk_indices[:n_test_chunks]
        train_chunk_indices = all_chunk_indices[n_test_chunks:]
        
        train_metadata = self._save_split(subsampled_data, train_output_file, input_file, 
                                         config, train_chunk_indices, "train")
        test_metadata = self._save_split(subsampled_data, test_output_file, input_file, 
                                        config, test_chunk_indices, "test")
        
        self.performance_metrics['split_and_save_time'] = time.time() - split_start_time
        
        return {
            "train": train_metadata,
            "test": test_metadata,
            "total_chunks": n_chunks,
            "subsample_rate": config.subsample_rate,
            "invariants_processed": list(subsampled_data['invariants'].keys())
        }

    def _validate_range(self, start_idx: Optional[int], end_idx: Optional[int], 
                       total_timesteps: int) -> Tuple[int, int]:
        start = 0 if start_idx is None else max(0, start_idx)
        end = total_timesteps if end_idx is None else min(total_timesteps, end_idx)
        
        if start >= end:
            raise ValueError(f"Invalid range: start ({start}) must be less than end ({end})")
        
        return start, end

    def _output_performance_report(self):
        print("\nPerformance Report:")
        for metric, value in self.performance_metrics.items():
            print(f"{metric}: {value:.2f} seconds")