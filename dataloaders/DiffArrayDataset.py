import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import os
import math
import random
import bitarray
import numpy as np

from binary_encoder import rle_decode_variable_length
from sklearn.utils.class_weight import compute_class_weight
from data_augmentation_experiments import reordering_bitarray

class BitArrayDataset(Dataset):
    def __init__(self, root_dir, folder):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        self.class_dict = {}
        self.folder = folder

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for folder_type in os.listdir(label_dir):
                    if folder_type == folder:
                        class_folder = os.path.join(label_dir, folder)
                        if label not in self.label_to_index:
                            index = len(self.label_to_index)
                            self.label_to_index[label] = index
                            self.index_to_label[index] = label
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

        # Load a sample to determine num_slices
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        self.num_slices = round(math.pow(len(ba_unpacked), 1 / 3))

        # Compute class weights using sklearn's approach
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
        anchor_tensor = self.load_bitarray(anchor_path)  # Returns NumPy array

        # Convert to tensor and move to GPU (if applicable)
        anchor_tensor = torch.tensor(anchor_tensor, dtype=torch.float16)  # Adjust dtype/device as needed

        # Perform reordering
        anchor_tensor = reordering_bitarray(anchor_tensor, self.num_slices)

        # Ensure correct shape
        anchor_tensor = anchor_tensor.view(self.num_slices, self.num_slices, self.num_slices)

        # Return class names along with tensors and integer labels
        return (anchor_tensor, anchor_label), self.num_slices


    def load_bitarray(self, file_path):
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, _, _ = rle_decode_variable_length(ba)
        ba_unpacked = np.frombuffer(ba.unpack(zero=b'\x00', one=b'\x01'), dtype=np.uint8)
        return ba_unpacked