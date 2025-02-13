import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.nn import CrossEntropyLoss

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from torch.nn.utils import weight_norm
from pytorch_metric_learning.miners import TripletMarginMiner, BatchHardMiner
from pytorch_metric_learning.losses import TripletMarginLoss, ContrastiveLoss
# from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.distances import LpDistance

from torch.nn.functional import pairwise_distance
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def harmonic_loss(predictions, targets):
    """
    Compute harmonic loss based on F1-score.

    Args:
        predictions: Logits of shape (batch_size, num_classes) BEFORE softmax.
        targets: Class indices of shape (batch_size).

    Returns:
        A scalar loss value (1 - mean F1-score).
    """
    epsilon = 1e-7  # To prevent division by zero

    # Convert predictions to probabilities
    predictions = F.softmax(predictions, dim=1)

    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1]).float()

    # Compute true positives, predicted positives, and actual positives per class
    true_positives = torch.sum(predictions * targets_one_hot, dim=0)  # Sum over batch
    predicted_positives = torch.sum(predictions, dim=0)  # Sum over batch
    actual_positives = torch.sum(targets_one_hot, dim=0)  # Sum over batch

    # Compute precision and recall with epsilon for numerical stability
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (actual_positives + epsilon)

    # Compute F1-score per class
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # Compute mean F1-score and return 1 - mean(F1) as loss
    return 1 - f1.mean()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels, dtype=torch.bfloat16)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv = self.conv.to(dtype=torch.bfloat16)
        self.shortcut = self.shortcut.to(dtype=torch.bfloat16)


    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)) + self.shortcut(x), negative_slope=0.01)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        batch_size, channels, length = x.size()
        se = x.mean(-1).view(batch_size, channels)  # Global Average Pooling
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(batch_size, channels, 1)  # Reshape for broadcasting
        return x * se

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        device = inputs.device
        targets = targets.to(device)
        self.weight = self.weight.to(device)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight).to(torch.bfloat16)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1, dtype=torch.bfloat16)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1, dtype=torch.bfloat16)  # Compute attention scores
        return (x * attn_weights).sum(dim=1)  # Weighted sum


class CustomCNN(pl.LightningModule):
    def __init__(self, num_classes, num_slices, dataset, alpha, gamma, margin, emb_dim, lr):
        super(CustomCNN, self).__init__()
        self.num_slices = num_slices
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.emb_dim = emb_dim
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
        self.test_embeddings = []
        self.class_weights = dataset.train_dataset.get_class_weights().to(self.device).to(torch.bfloat16)
        self.lr = lr
        self.miner = BatchHardMiner()


        # # Existing convolutional layers
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, out_channels=16, kernel_size=1)
        self.conv3 = nn.Conv1d(16, 32, 1)
        self.conv4 = nn.Conv1d(32, 64, 1)
        self.conv5 = nn.Conv1d(64, 64, 1)
        self.conv6 = nn.Conv1d(64, 128, 1)
        self.conv7 = nn.Conv1d(128, 256, 1)


        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(256)

        # Convert layers to bfloat16 for performance
        # for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
        #     layer.to(torch.bfloat16)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveMaxPool1d(16)

        self.fc1 = nn.Linear(256 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.emb_dim)
        self.fc_bn2 = nn.BatchNorm1d(self.emb_dim)
        self.fc3 = nn.Linear(self.emb_dim, self.num_classes)

        self.dropout1 = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.8)

        # Loss (harmonic loss is implemented as a function)
        self.triplet_loss = ContrastiveLoss()


    def forward(self, x, embeddings=False):
        batch_size, _, _ = x.size()
        # Reshape input to match the new channel size for Conv1D

        # Apply Conv1D layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        # Global pooling and fully connected layers
        pool = self.global_pool(x)  # Shape: (batch_size, channels)
        flattened_output = pool.view(batch_size, -1)
        
        output = F.relu(self.fc_bn1(self.fc1(flattened_output)))
        output = self.dropout1(output)

        embedding = F.relu(self.fc_bn2(self.fc2(output)))  # Embedding layer

        output = self.dropout2(embedding)
        output = self.fc3(output)
        return output, embedding

    
    def steps(self, anchor, type, batch_size):
        anchor_input, anchor_label, _ = anchor

        ## Forward pass
        anchor_output, anchor_embedding = self(anchor_input)         # Anchor is the sample being trained on
        ## Loss Functions
        # loss_classification = self.classification_loss(anchor_output, anchor_label)
        loss_classification = harmonic_loss(anchor_output, anchor_label)

        # Run the miner
        hard_pairs = self.miner(anchor_embedding, anchor_label)

        loss_triplet = self.triplet_loss(anchor_embedding, anchor_label, hard_pairs)
        loss = loss_classification + 0.5 * loss_triplet  # Weighting losses
        loss = loss_classification


        ## Accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        ## Logging
        self.steps_log(loss, loss_classification, loss_triplet, acc, preds, type, batch_size)

        ## Storing for future evals
        if type == "val":
            self.val_preds.append(preds)
            self.val_labels.append(anchor_label)

        if type == "test":
            _, embedding = self(anchor_input, True)
            self.test_preds.append(preds)
            self.test_labels.append(anchor_label)
            self.test_embeddings.append(embedding)

        return loss

    def steps_log(self, loss, loss_classification, loss_triplet, acc, preds, type, batch_size):
        if type == "train":
            self.log(f'Loss/{type}_loss', loss, prog_bar=True, batch_size=batch_size)
            self.log(f'Loss/{type}_class_loss', loss_classification, prog_bar=False, batch_size=batch_size)
            self.log(f'Loss/{type}_trip_loss', loss_triplet, prog_bar=False, batch_size=batch_size)
            self.log(f'Acc/{type}_acc', acc, prog_bar=True, batch_size=batch_size)
        else:
            self.log(f'Loss/{type}_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f'Loss/{type}_class_loss', loss_classification, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log(f'Loss/{type}_trip_loss', loss_triplet, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log(f'Acc/{type}_acc', acc, prog_bar=True, batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx):
        anchor_tensor,  _ = batch
        loss = self.steps(anchor_tensor, "train", self.trainer.datamodule.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_tensor, _ = batch
        loss = self.steps(anchor_tensor, "val", self.trainer.datamodule.batch_size)

        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_tensor, _ = batch
        loss = self.steps(anchor_tensor, "test", self.trainer.datamodule.batch_size)

        return loss

    def on_val_epoch_end(self):
        with torch.no_grad():
            # Compute confusion matrix
            all_preds = torch.cat(self.val_preds)
            all_labels = torch.cat(self.val_labels)

            cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(self.num_classes))
            class_names = list(self.trainer.datamodule.val_dataset.label_to_index.keys())  # Fetch class names
            self.log_confusion_matrix(cm, "Validation", class_names)

            # Clear the stored predictions and labels
            self.val_preds.clear()
            self.val_labels.clear()

    def on_test_epoch_end(self):
        with torch.no_grad():
            # Fetch class names from the test dataset
            class_names = list(self.trainer.datamodule.test_dataset.label_to_index.keys())
            
            all_preds = torch.cat(self.test_preds)
            all_labels = torch.cat(self.test_labels)
            all_embeddings = torch.cat(self.test_embeddings)

            # Ensure consistent length
            assert len(all_labels) == len(all_preds), "Length mismatch between labels and predictions."

            cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(self.num_classes))
            self.log_confusion_matrix(cm, "Test", class_names)

            self.log_embeddings(all_labels, all_embeddings, class_names)

            # Clear the stored test embeddings and labels
            self.test_preds.clear()
            self.test_embeddings.clear()
            self.test_labels.clear()


    
    def log_confusion_matrix(self, cm, name, class_names):
        with torch.no_grad():
            # Format class names for y-axis as "Class Name (Numerical Label)"
            true_label_names = [f"{class_name} ({i})" for i, class_name in enumerate(class_names)]
            
            # Calculate total samples for each class (sum over the rows of the confusion matrix)
            total_samples = cm.sum(axis=1)

            # Create a confusion matrix plot with custom labels
            fig, ax = plt.subplots(figsize=(20, 20))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(class_names)))  # Use numerical labels for x-axis
            disp.plot(ax=ax, cmap="Blues", colorbar=False)  # Remove default colorbar

            # Set the x-axis (Predicted Labels) as numerical
            ax.set_xlabel("Predicted Labels (Numerical)", fontsize=12)
            
            # Set the y-axis (True Labels) with custom formatted labels
            ax.set_yticklabels(true_label_names, fontsize=10)
            ax.set_ylabel("True Labels (Class Name + Numerical)", fontsize=12)

            # Add total sample counts to the right y-axis
            ax2 = ax.twinx()  # Create a second y-axis that shares the same x-axis
            ax2.set_ylim(ax.get_ylim())  # Ensure alignment with the left y-axis
            ax2.set_yticks(ax.get_yticks())
            ax2.set_yticklabels([f'{total}' for total in total_samples], fontsize=10)
            ax2.set_ylabel('Total Samples', fontsize=12)

            # Set the title
            plt.title(f'{name} Confusion Matrix', fontsize=14)
            # plt.tight_layout()

            # Save the figure
            plt.savefig("tb_logs/" + self.logger.name + "/version_" + str(self.logger.version) + "/" +  name + "_confusion_matrix.png")
            plt.close(fig)

            # Log the confusion matrix image to TensorBoard (uncomment if needed)
            # self.logger.experiment.add_image("Confusion Matrix", 
            #                                   self.tensorboard_image_from_figure(fig), 
            #                                   self.current_epoch)

    
    def log_embeddings(self, labels, embeddings, class_names):
        with torch.no_grad():
            # Convert numerical labels to string class names
            string_labels = [class_names[label.item()] for label in labels]
            embeddings = embeddings.to(torch.float32)
            
            # Log the embeddings to TensorBoard with string labels
            self.logger.experiment.add_embedding(embeddings, metadata=string_labels, global_step=self.current_epoch)


    @staticmethod
    def tensorboard_image_from_figure(fig):
        """Convert a matplotlib figure to a tensorboard image."""
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5, cooldown=3, min_lr=1e-5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Loss/val_loss'}