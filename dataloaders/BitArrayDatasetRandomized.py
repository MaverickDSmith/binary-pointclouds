import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import pytorch_lightning as pl

import os
import math
import random
import bitarray
import numpy as np

from binary_encoder import rle_decode_variable_length
from sklearn.utils.class_weight import compute_class_weight


import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, SequentialSampler
import bitarray
import numpy as np
import random
import math
from sklearn.utils.class_weight import compute_class_weight

class BitArrayDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.class_dict = {}

        for label in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                label_index = len(self.label_to_index)
                self.label_to_index[label] = label_index
                self.class_dict[label_index] = []
                for file_name in sorted(os.listdir(label_dir)):
                    if file_name.endswith('.bin'):
                        file_path = os.path.join(label_dir, file_name)
                        self.file_paths.append(file_path)
                        self.labels.append(label_index)
                        self.class_dict[label_index].append(file_path)

        # Load a sample to determine num_slices (assume all files have same dimensions)
        with open(self.file_paths[0], 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length(ba)
        self.num_slices = round(math.pow(len(ba.unpack()), 1 / 3))

        # Compute class weights
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.labels), y=self.labels)

    def get_class_weights(self):
        return torch.tensor(self.class_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load anchor
        print(idx)
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]
        anchor_tensor = self.load_and_process(anchor_path)
        
        # Load a positive sample from the same class
        positive_idx = self.get_positive_sample(anchor_label, idx)
        positive_path = self.file_paths[positive_idx]
        positive_tensor = self.load_and_process(positive_path)
        
        # Load a negative sample from a different class
        negative_idx = self.get_negative_sample(anchor_label)
        negative_path = self.file_paths[negative_idx]
        negative_tensor = self.load_and_process(negative_path)
        
        return (anchor_tensor, anchor_label), (positive_tensor, self.labels[positive_idx]), (negative_tensor, self.labels[negative_idx]), self.num_slices

    def get_positive_sample(self, anchor_label, anchor_idx):
        """Get a positive sample index, avoiding the anchor index."""
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != anchor_idx]
        return random.choice(positive_indices)

    def get_negative_sample(self, anchor_label):
        """Get a negative sample index."""
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        return random.choice(negative_indices)



    def load_and_process(self, file_path):
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)  # Read the entire file at once
        ba, _, _ = rle_decode_variable_length(ba)
        
        # Convert to NumPy and immediately free the bitarray
        ba_unpacked = np.frombuffer(ba.unpack(), dtype=np.uint8)
        del ba  # Free the bitarray object explicitly
        
        # Convert to PyTorch tensor and reshape
        tensor = torch.tensor(ba_unpacked, dtype=torch.bfloat16)
        return tensor.view(self.num_slices, self.num_slices, self.num_slices)



class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, num_workers=0, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        # Seed everything for reproducibility
        pl.seed_everything(self.seed, workers=True)

        # Initialize the dataset
        self.full_dataset = BitArrayDataset(self.root_dir)

        # Calculate split sizes
        total_size = len(self.full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size  # Ensures no files are left out

        # Split the dataset
        train_indices, val_indices, test_indices = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)  # For reproducibility
        )

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SequentialSampler(val_indices)
        self.test_sampler = SequentialSampler(test_indices)
        

    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, sampler=self.test_sampler, num_workers=self.num_workers, pin_memory=True)

