import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pytorch_lightning as pl
import os
import random
import bitarray
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import math
# Assuming you have the existing decoder methods
from binary_encoder import rle_decode_variable_length, sc_decode_variable_length_with_bounds, rle_decode_variable_length_voxels
from experiments.data_augmentation_experiments import reordering_bitarray

from collections import Counter

class BitArrayDataset(Dataset):
    def __init__(self, root_dir, split="train", split_ratios=(0.8, 0.2), seed=42):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        self.class_dict = {}

        # Iterate through each class in the root directory
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                # Map class labels to indices
                if label not in self.label_to_index:
                    index = len(self.label_to_index)
                    self.label_to_index[label] = index
                    self.index_to_label[index] = label

                # Handle train/test subdirectories for each class
                if split == "train" or split == "val":
                    split_name = "train"
                else:
                    split_name = "test"
                split_dir = os.path.join(label_dir, split_name)
                if os.path.isdir(split_dir):
                    # Debugging: Check if we have files in this split
                    found_files = False
                    for file_name in os.listdir(split_dir):
                        if file_name.endswith('.bin'):
                            found_files = True
                            file_path = os.path.join(split_dir, file_name)
                            self.file_paths.append(file_path)
                            self.labels.append(self.label_to_index[label])

                            # Add to class_dict for triplet sampling
                            label_index = self.label_to_index[label]
                            if label_index not in self.class_dict:
                                self.class_dict[label_index] = []
                            self.class_dict[label_index].append(file_path)

                    if not found_files:
                        print(f"Warning: No .bin files found in {split_dir} for class {label}")

        # If no files were found, raise an error to notify that something went wrong
        if len(self.file_paths) == 0:
            raise ValueError(f"No .bin files found in the {split} splits for any class.")

        # Shuffle and split into train/val (if not empty)
        random.seed(seed)
        combined = list(zip(self.file_paths, self.labels))
        random.shuffle(combined)
        self.file_paths, self.labels = zip(*combined)

        num_samples = len(self.file_paths)
        train_end = int(num_samples * split_ratios[0])
        val_end = train_end + int(num_samples * split_ratios[1])

        if split == "train":
            self.file_paths = self.file_paths[:train_end]
            self.labels = self.labels[:train_end]
        elif split == "val":
            self.file_paths = self.file_paths[train_end:val_end]
            self.labels = self.labels[train_end:val_end]

        # Load a sample to determine num_slices
        ba = bitarray.bitarray()
        with open(self.file_paths[0], 'rb') as f:
            ba = f.read()
        ba, _, _ = sc_decode_variable_length_with_bounds(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        self.num_slices = round(math.pow(len(ba_unpacked), 1 / 3))

        # Compute class weights
        labels_array = np.array(self.labels)
        unique_labels = np.arange(len(self.label_to_index))
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels_array)
        self.class_weights = class_weights

    def get_class_weights(self):
        return torch.tensor(self.class_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load anchor
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]
        anchor_tensor = self.load_bitarray(anchor_path)
        anchor_class_name = self.index_to_label[anchor_label]
        # Convert to tensors and reshape
        anchor_tensor = torch.tensor(anchor_tensor, dtype=torch.float32)

        # Randomly decide whether to apply rotation
        # if random.random() < 0.5:  # 50% chance to rotate
        #     anchor_tensor = reordering_bitarray(anchor_tensor, self.num_slices)
        # else:
        # anchor_tensor = anchor_tensor.view(self.num_slices, self.num_slices, self.num_slices)
        # anchor_tensor = anchor_tensor.permute(2, 1, 0).contiguous()
        # anchor_tensor = anchor_tensor.view(-1)
        anchor_tensor = anchor_tensor.view(self.num_slices, self.num_slices, self.num_slices)

        return (anchor_tensor, anchor_label, anchor_class_name), \
               self.num_slices

    def load_bitarray(self, file_path):
        ba = bitarray.bitarray()
        with open(file_path, 'rb') as f:
            ba = f.read()
        ba, _, _ = sc_decode_variable_length_with_bounds(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        return ba_unpacked

# class PointCloudDataModule(pl.LightningDataModule):
#     def __init__(self, root_dir, batch_size=24, num_workers=16, split_ratios=(0.8, 0.2), seed=42):
#         super().__init__()
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.split_ratios = split_ratios
#         self.seed = seed
#         self.weights = None
#         self.sampler = None

#     def setup(self, stage=None):
#         # Setup for train/val datasets with a split for the Train folders
#         self.train_dataset = BitArrayDataset(self.root_dir, split="train", split_ratios=self.split_ratios, seed=self.seed)
#         self.val_dataset = BitArrayDataset(self.root_dir, split="val", split_ratios=self.split_ratios, seed=self.seed)
#         # Setup for test dataset (not split)
#         self.test_dataset = BitArrayDataset(self.root_dir, split="test", split_ratios=(1.0, 0.0), seed=self.seed)

#         # Create class weights and sample weights
#         self.class_weights = self.train_dataset.get_class_weights()
#         sample_weights = [self.class_weights[label] for label in self.train_dataset.labels]
        
#         # Setup the WeightedRandomSampler
#         self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, 
#                           num_workers=self.num_workers, pin_memory=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
#                           num_workers=self.num_workers, pin_memory=True)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
#                           num_workers=self.num_workers, pin_memory=True)



class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, num_workers=16, split_ratios=(0.8, 0.2), seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.seed = seed
        self.weights = None
        self.sampler = None

    def setup(self, stage=None):
        # Setup for train/val datasets with a split for the Train folders
        self.train_dataset = BitArrayDataset(self.root_dir, split="train", split_ratios=self.split_ratios, seed=self.seed)
        self.val_dataset = BitArrayDataset(self.root_dir, split="val", split_ratios=self.split_ratios, seed=self.seed)
        # Setup for test dataset (not split)
        self.test_dataset = BitArrayDataset(self.root_dir, split="test", split_ratios=(1.0, 0.0), seed=self.seed)

        # # Create class weights and sample weights
        # self.class_weights = self.train_dataset.get_class_weights()
        # sample_weights = [self.class_weights[label] for label in self.train_dataset.labels]
        
        # # Setup the WeightedRandomSampler
        # self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, 
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)
