import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import bitarray
import numpy as np
import math
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from binary_encoder import rle_decode_variable_length
import random

import sys


class BitArrayDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.class_dict = {}

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                if label not in self.label_to_index:
                    self.label_to_index[label] = len(self.label_to_index)
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.bin'):
                        file_path = os.path.join(label_dir, file_name)
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
        anchor_tensor, _, _ = rle_decode_variable_length(ba)
        self.extra_bits = anchor_tensor.fill()
        print(self.extra_bits)
        self.num_slices = round(math.pow(len(anchor_tensor), 1/3))
        self.numpoints = len(anchor_tensor)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load anchor
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]
        anchor_tensor = self.load_bitarray(anchor_path)
        
        # Load a positive sample from the same class
        positive_idx = self.get_positive_sample(anchor_label)
        positive_path = self.file_paths[positive_idx]
        positive_tensor = self.load_bitarray(positive_path)
        
        # Load a negative sample from a different class
        negative_idx = self.get_negative_sample(anchor_label)
        negative_path = self.file_paths[negative_idx]
        negative_tensor = self.load_bitarray(negative_path)

        # anchor_tensor = torch.tensor(anchor_tensor, dtype=torch.uint8)
        # positive_tensor = torch.tensor(positive_tensor, dtype=torch.uint8)
        # negative_tensor = torch.tensor(negative_tensor, dtype=torch.uint8)

        # Convert to tensors and return as a tuple
        return (anchor_tensor, anchor_label), \
               (positive_tensor, self.labels[positive_idx]), \
               (negative_tensor, self.labels[negative_idx]), \
               self.num_slices, \
               self.extra_bits

    def load_bitarray(self, file_path):
            with open(file_path, 'rb') as f:
                ba = bitarray.bitarray()
                ba.fromfile(f)
            ba, _, _ = rle_decode_variable_length(ba)
            # return np.packbits(np.frombuffer(ba, dtype=np.uint8))
            # _ = ba.fill()
            ba_packed = np.packbits(np.frombuffer(ba, dtype=np.uint8))
            # print(f"BA Length: {len(ba)}")
            # print(f"BA Shape: {np.shape(ba)}")
            # print(f"BA Packed Length: {len(ba_packed)}")
            # print(f"BA Packed Shape: {np.shape(ba_packed)}")
            return ba_packed

    def get_positive_sample(self, anchor_label):
        """Get a positive sample index for the given label."""
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label]
        return random.choice(positive_indices)

    def get_negative_sample(self, anchor_label):
        """Get a negative sample index different from the anchor label."""
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        return random.choice(negative_indices)
    

# LightningModule that handles the model and training/validation loop
class CustomCNN(pl.LightningModule):
    def __init__(self, num_classes, num_slices):
        super(CustomCNN, self).__init__()
        self.num_slices = num_slices
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool2d((4, 4))

        self.fc1 = nn.Linear(64 * num_slices * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(256)

        # Losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)



## TODO: Make sure this forward function is actually iterating over the packed bits correctly
## Currently having issues because the dimensions aren't what is expected for self.fc1 and what the output of the conv layers are
## Slicing logic is different now so it's gotta be figured out
    def forward(self, x, num_slices, extra_bits):
        # x: input in numpy bytes format (e.g., packed bits)
        # num_slices: number of slices (e.g., 65 slices per object)
        # extra_bits: the extra bits to be removed from unpacked bits (for each object)

        batch_size, num_points = x.size()
        
        # Convert numpy array `x` back to `bitarray`
        bit_sequences = []
        
        for i in range(batch_size):
            # Convert each row of x (object_in_bytes) to bitarray
            ba = bitarray.bitarray(endian='big')
            ba.frombytes(np.unpackbits(x[i].cpu(), count=-extra_bits[0]))  # Assuming x is a numpy array of bytes
            # print(np.shape(ba))
            bit_sequences.append(ba)

        outputs = []
        l = num_slices[0] * num_slices[0]  # Adjust for the number of slices (e.g., 65 * 65)
        counter = 0
        
        for slice_idx in range(num_slices[0]):
            # Process each object in the batch
            for batch_idx, bit_seq in enumerate(bit_sequences):
                # Extract bits for the current slice
                slice_bits = bit_seq[counter:l]
                print(counter)

                # Convert bitarray slice back to numpy array for PyTorch processing
                slice_tensor = torch.tensor(np.frombuffer(slice_bits.tobytes(), dtype=np.uint8), device=x.device, dtype=torch.uint8)
                
                # Unpack bits into 2D form for convolutional layers
                unpacked_bits = ((slice_tensor.unsqueeze(-1) >> torch.arange(8, device=x.device)) & 1).float()
                
                # Reshape unpacked bits for conv layers (batch_size, channels, height, width)
                bit_tensor = unpacked_bits.view(batch_size, 1, 1, 8)  # shape: (batch_size, 1, 8, 1)
                
                # Pass through convolutional layers
                conv_output = F.relu(self.bn1(self.conv1(bit_tensor)))
                conv_output = F.relu(self.bn2(self.conv2(conv_output)))
                
                outputs.append(conv_output)

            counter = l + 1
            l = l + (num_slices[0] * num_slices[0])  # Adjust to next slice

        # Concatenate outputs across slices
        fused_output = torch.cat(outputs, dim=1)
        
        # Global pooling and fully connected layers
        fused_output = self.global_pool(fused_output)
        flattened_output = fused_output.view(batch_size, -1)

        # Final classification layer
        fc_output = self.dropout(F.relu(self.fc1(flattened_output)))
        output = self.fc2(fc_output)

        return output

    # def forward(self, x, num_slices, extra_bits):
    #     batch_size, num_points = x.size()
    #     # print(np.shape(num_points))
    #     # print(np.shape(x))
    #     # print(x)
    #     outputs = []
    #     l = num_slices[0] * num_slices[0]
    #     counter = 0
    #     x[1] = np.unpackbits(x[1], dtype=np.uint8)
    #     x[1].to(torch.tensor(dtype=torch.float32))
    #     for slice_idx in range(num_slices[0]):
    #         # Get the packed slice (one byte, 8 bits per element)
    #         packed_slice_input = x[:, slice_idx:slice_idx+1] # Ensure it's a byte tensor
    #         # print(packed_slice_input)
    #         # print(np.shape(packed_slice_input))
    #         # Manually unpack the bits from the byte
    #         # torch does not have unpackbits, so we simulate it
    #         unpacked_bits = ((packed_slice_input.unsqueeze(-1) >> torch.arange(8, device=x.device)) & 1).float()
    #         print(unpacked_bits)
    #         print(np.shape(unpacked_bits))
    #         # unpacked_bits is now shape: [batch_size, 8] (8 bits for each packed byte)

    #         # Pass the bit_tensor through the conv layers
    #         bit_tensor = unpacked_bits.view(batch_size, 1, 1, 8)  # shape: (batch_size, 1, 8, 1)
    #         conv_output = F.relu(self.bn1(self.conv1(bit_tensor)))
    #         conv_output = F.relu(self.bn2(self.conv2(conv_output)))
    #         outputs.append(conv_output)
    #         counter = l + 1
    #         l = l + (num_slices[0] * num_slices[0])

    #     # Concatenate outputs across the slice dimension (dim=1)
    #     fused_output = torch.cat(outputs, dim=1)
    #     # print(np.shape(fused_output))
        
    #     # Global pooling layer and fully connected layers
    #     fused_output = self.global_pool(fused_output)
    #     flattened_output = fused_output.view(batch_size, -1)

    #     fc_output = self.dropout(F.relu(self.fc1(flattened_output)))
    #     output = self.fc2(fc_output)

    #     return output


    # def forward(self, x, num_slices, extra_bits):
    #     batch_size, num_points = x.size()
    #     outputs = []

    #     bit_counter = 0  # This will track where we are in terms of bits
    #     bits_per_slice = 4225  # 65^2 bits per slice
    #     bits_in_byte = 8
    #     counter = 0

    #     for slice_idx in range(num_slices[0]):
    #         # Initialize a list to accumulate bits for each batch (keep batch dimension intact)
    #         slice_bits = torch.zeros(batch_size, bits_per_slice, device=x.device)  # Shape: (batch_size, 4225)
    #         slice_bit_index = 0  # This will track how many bits we've added to the slice

    #         # Continue pulling bytes from the byte arrays in x until we accumulate enough bits
    #         while slice_bit_index < bits_per_slice:
    #             # Calculate how many bits are remaining in the current byte array
    #             current_byte_idx = bit_counter // bits_in_byte  # Integer division to get the current byte index
    #             bit_offset = bit_counter % bits_in_byte  # Modulo to get the bit offset within the byte

    #             # Ensure we don't exceed the number of byte arrays available in x
    #             if current_byte_idx >= num_points:
    #                 raise RuntimeError("Exceeded the available number of byte arrays in x")

                
    #             # Get the current byte array for the batch
    #             packed_byte_array = x[:, current_byte_idx]  # Shape: (batch_size,)
                

    #             # Unpack the bits from this byte (starting from the bit offset)
    #             unpacked_bits = ((packed_byte_array.unsqueeze(-1) >> torch.arange(bits_in_byte, device=x.device)) & 1)

    #             # Determine how many bits to take from this byte (up to 8 bits, but may be fewer if close to the target)
    #             bits_to_take = min(bits_per_slice - slice_bit_index, bits_in_byte - bit_offset)

    #             # Add bits from the current byte to the slice, starting at bit_offset and adding `bits_to_take` bits
    #             slice_bits[:, slice_bit_index:slice_bit_index + bits_to_take] = unpacked_bits[:, bit_offset:bit_offset + bits_to_take]

    #             # Update the counters
    #             bit_counter += bits_to_take
    #             slice_bit_index += bits_to_take
    #             counter += 1

    #         # At this point, slice_bits contains exactly 4225 bits for each batch
    #         # Reshape it into a tensor suitable for passing through the convolution layers
    #         bit_tensor = slice_bits.view(batch_size, 1, 1, bits_per_slice)  # Shape: (batch_size, 1, 1, 4225)

    #         # Pass through convolutional layers
    #         conv_output = F.relu(self.bn1(self.conv1(bit_tensor)))
    #         conv_output = F.relu(self.bn2(self.conv2(conv_output)))

    #         # Store the output of the conv layers
    #         outputs.append(conv_output)

    #     # Concatenate outputs across the slice dimension (dim=1)
    #     fused_output = torch.cat(outputs, dim=1)

    #     # Global pooling layer and fully connected layers
    #     fused_output = self.global_pool(fused_output)
    #     flattened_output = fused_output.view(batch_size, -1)

    #     fc_output = self.dropout(F.relu(self.fc1(flattened_output)))
    #     output = self.fc2(fc_output)

    #     return output


    def training_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, num_slices, extra_bits = batch
        anchor_input, anchor_label = anchor_tensor
        positive_input, _ = positive_tensor
        negative_input, _ = negative_tensor

        # Ensure efficient unpacking before forward pass
        anchor_output = self(anchor_input, num_slices, extra_bits)
        positive_output = self(positive_input, num_slices, extra_bits)
        negative_output = self(negative_input, num_slices, extra_bits)

        loss_classification = self.classification_loss(anchor_output, anchor_label)
        
        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        loss = loss_classification + 0.5 * loss_triplet  # Weighting losses

        self.log('Loss/train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, num_slices, extra_bits = batch
        anchor_input, anchor_label = anchor_tensor
        positive_input, _ = positive_tensor
        negative_input, _ = negative_tensor

        # Ensure efficient unpacking before forward pass
        anchor_output = self(anchor_input, num_slices, extra_bits)
        positive_output = self(positive_input, num_slices, extra_bits)
        negative_output = self(negative_input, num_slices, extra_bits)

        # Ensure efficient unpacking before forward pass
        loss_classification = self.classification_loss(anchor_output, anchor_label)
        
        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        loss = loss_classification + 0.5 * loss_triplet  # Weighting losses

        # Calculate accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        self.log('Loss/val_loss', loss, prog_bar=True)
        self.log('Acc/val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, num_slices, extra_bits = batch
        anchor_input, anchor_label = anchor_tensor
        positive_input, _ = positive_tensor
        negative_input, _ = negative_tensor

        # Ensure efficient unpacking before forward pass
        anchor_output = self(anchor_input, num_slices, extra_bits)
        positive_output = self(positive_input, num_slices, extra_bits)
        negative_output = self(negative_input, num_slices, extra_bits)
        loss_classification = self.classification_loss(anchor_output, anchor_label)
        
        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        loss = loss_classification + 0.5 * loss_triplet  # Weighting losses

        # Calculate accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        self.log('Loss/test_loss', loss, prog_bar=True)
        self.log('Acc/test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Loss/val_loss'}



# DataModule that handles loading the data
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, val_size=0.2, test_size=0.1, num_workers=24):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = BitArrayDataset(self.root_dir)
        train_size = int((1 - self.val_size - self.test_size) * len(dataset))
        val_size = int(self.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)




# Instantiate the DataModule and Model
sys.set_int_max_str_digits(274625)
root_dir = '/home/hi5lab/pointcloud_data/storage_test_two/slice64'
datamodule = PointCloudDataModule(root_dir)
datamodule.setup()
_, _, _, num_slices, _ = next(iter(datamodule.train_dataloader()))
num_slices = num_slices[0].item()
model = CustomCNN(num_classes=40, num_slices=num_slices)

# Logging and Checkpointing
logger = TensorBoardLogger("tb_logs", name="pointcloud_cnn")
checkpoint_callback = ModelCheckpoint(monitor='Acc/val_acc', save_top_k=1, mode='min', filename='{epoch}-{val_loss:.4f}')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Train the model using PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=50,
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    accelerator='gpu',  # Ensure you're using GPUs
    devices=1,  # Set to the number of GPUs you want to use (1 for single-GPU)
    strategy=DDPStrategy(find_unused_parameters=False),  # Enable DDP
    num_nodes=1  # This will be more than 1 when scaling to multiple GPUs
)
trainer.fit(model, datamodule)

# Test the model
trainer.test(model, datamodule.test_dataloader())
