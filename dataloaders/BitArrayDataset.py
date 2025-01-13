import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

import os
import math
import random
import bitarray
import numpy as np

from binary_encoder_test import rle_decode_variable_length_test
from sklearn.utils.class_weight import compute_class_weight

class BitArrayDataset(Dataset):
    def __init__(self, root_dir, split="train", split_ratios=(0.7, 0.2, 0.1), seed=42):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        self.class_dict = {}

        # Collect all file paths and their labels
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                if label not in self.label_to_index:
                    index = len(self.label_to_index)
                    self.label_to_index[label] = index
                    self.index_to_label[index] = label
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.bin'):
                        file_path = os.path.join(label_dir, file_name)
                        self.file_paths.append(file_path)
                        self.labels.append(self.label_to_index[label])

                        # Add to class_dict for triplet sampling
                        label_index = self.label_to_index[label]
                        if label_index not in self.class_dict:
                            self.class_dict[label_index] = []
                        self.class_dict[label_index].append(file_path)

        # Split data into train/val/test
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
        elif split == "test":
            self.file_paths = self.file_paths[val_end:]
            self.labels = self.labels[val_end:]

        # Load a sample to determine num_slices
        with open(self.file_paths[0], 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length_test(ba)
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

        # Load a positive sample from the same class
        positive_idx = self.get_positive_sample(anchor_label)
        positive_tensor = self.load_bitarray(self.file_paths[positive_idx])

        # Load a negative sample from a different class
        negative_idx = self.get_negative_sample(anchor_label)
        negative_tensor = self.load_bitarray(self.file_paths[negative_idx])

        # Convert to tensors and reshape
        anchor_tensor = torch.tensor(anchor_tensor, dtype=torch.bfloat16)
        # print(anchor_path, len(anchor_tensor))
        anchor_tensor = anchor_tensor.view(self.num_slices, self.num_slices, self.num_slices)
        positive_tensor = torch.tensor(positive_tensor, dtype=torch.bfloat16)
        # print(self.file_paths[positive_idx], len(positive_tensor))
        positive_tensor = positive_tensor.view(self.num_slices, self.num_slices, self.num_slices)
        negative_tensor = torch.tensor(negative_tensor, dtype=torch.bfloat16)
        negative_tensor = negative_tensor.view(self.num_slices, self.num_slices, self.num_slices)

        return (anchor_tensor, anchor_label, anchor_class_name), \
               (positive_tensor, self.labels[positive_idx]), \
               (negative_tensor, self.labels[negative_idx]), \
               self.num_slices

    def load_bitarray(self, file_path):
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length_test(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        return ba_unpacked

    def get_positive_sample(self, anchor_label):
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label]
        return random.choice(positive_indices)

    def get_negative_sample(self, anchor_label):
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        return random.choice(negative_indices)


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, num_workers=16, split_ratios=(0.7, 0.2, 0.1), seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = BitArrayDataset(self.root_dir, split="train", split_ratios=self.split_ratios, seed=self.seed)
        self.val_dataset = BitArrayDataset(self.root_dir, split="val", split_ratios=self.split_ratios, seed=self.seed)
        self.test_dataset = BitArrayDataset(self.root_dir, split="test", split_ratios=self.split_ratios, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
