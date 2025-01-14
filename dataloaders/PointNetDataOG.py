import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split, split_ratios=(0.7, 0.2, 0.1), seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_ratios = split_ratios
        self.file_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))  # Sorted for consistent indexing
        np.random.seed(seed)

        # Gather all file paths and their corresponding labels
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name, split)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.off'):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label_idx)

        # Shuffle and split the data if needed
        data = list(zip(self.file_paths, self.labels))
        np.random.shuffle(data)
        self.file_paths, self.labels = zip(*data)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        points = self._read_off_file(file_path)
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _read_off_file(file_path):
        """Reads an .off file and extracts point cloud data."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines[0].strip() != "OFF":
                raise ValueError(f"{file_path} is not a valid .off file")
            # Extract number of vertices
            n_vertices = int(lines[1].split()[0])
            # Read points (vertices)
            points = [list(map(float, line.split())) for line in lines[2:2 + n_vertices]]
        return np.array(points)


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, num_workers=16, split_ratios=(0.7, 0.2, 0.1), seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = PointCloudDataset(self.root_dir, split="train", split_ratios=self.split_ratios, seed=self.seed)
        self.val_dataset = PointCloudDataset(self.root_dir, split="val", split_ratios=self.split_ratios, seed=self.seed)
        self.test_dataset = PointCloudDataset(self.root_dir, split="test", split_ratios=self.split_ratios, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
