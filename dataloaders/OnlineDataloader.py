import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import os
import math
import random
import bitarray
import numpy as np

from binary_encoder import rle_decode_variable_length

class BitArrayDataset(Dataset):
    def __init__(self, root_dir, folder):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.class_dict = {}
        self.folder = folder

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for folder_type in os.listdir(label_dir):
                    if folder_type == folder:
                        class_folder = os.path.join(label_dir, folder)
                        if label not in self.label_to_index:
                            self.label_to_index[label] = len(self.label_to_index)
                        for file_name in os.listdir(class_folder):
                            if file_name.endswith('.bin'):
                                file_path = os.path.join(class_folder, file_name)
                                self.file_paths.append(file_path)
                                label_index = self.label_to_index[label]
                                self.labels.append(label_index)

                                # Add to class_dict for triplet sampling
                                if label_index not in self.class_dict:
                                    self.class_dict[label_index] = []
                                self.class_dict[label_index].append(file_path)
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        self.num_slices = round(math.pow(len(ba_unpacked), 1/3))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load anchor
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]
        anchor_tensor = self.load_bitarray(anchor_path)
    
        num_slices = round(math.pow(len(anchor_tensor), 1/3))
        anchor_tensor = torch.tensor(anchor_tensor, dtype=torch.bfloat16)
        anchor_tensor = anchor_tensor.view(num_slices, num_slices, num_slices)

        # Convert to tensors and return as a tuple
        return (anchor_tensor, anchor_label), \
               num_slices

    def load_bitarray(self, file_path):
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        return ba_unpacked
    
    def get_positive_sample(self, anchor_label):
        """Get a positive sample index for the given label."""
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label]
        return random.choice(positive_indices)

    def get_negative_sample(self, anchor_label):
        """Get a negative sample index different from the anchor label."""
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        return random.choice(negative_indices)

# DataModule that handles loading the data
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, num_workers=16):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BitArrayDataset(self.root_dir, "train")
        self.test_dataset  = BitArrayDataset(self.root_dir, "test")
        self.val_dataset   = BitArrayDataset(self.root_dir, "val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
