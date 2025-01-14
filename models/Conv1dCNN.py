import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from torch.nn.utils import weight_norm


# class FocalLoss(pl.LightningModule):
#     def __init__(self, alpha=0.5, gamma=1.5, weight=None, reduction='mean'):
#         """
#         :param alpha: Weighting factor to balance the importance of positive/negative examples
#         :param gamma: Focusing parameter to reduce the relative loss for well-classified examples
#         :param weight: Class weights (e.g., computed from class distribution)
#         :param reduction: 'mean' or 'sum' to reduce the loss to a scalar
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def forward(self, input, target):
#         """
#         :param input: Predictions, of shape (batch_size, num_classes)
#         :param target: Ground truth labels, of shape (batch_size)
#         :return: Focal loss
#         """
#         device = input.device  # Ensure all tensors are on the same device
#         target = target.to(device)
#         self.weight = self.weight.to(device)

#         # Get log probabilities (log-softmax)
#         log_pt = F.log_softmax(input, dim=-1).to(device)  # Move log probabilities to correct device
#         pt = torch.exp(log_pt)  # Shape: (batch_size, num_classes)

#         # Gather the log probabilities for the correct class labels
#         log_pt = log_pt.gather(dim=-1, index=target.unsqueeze(-1).to(device))  # Shape: (batch_size, 1)
#         log_pt = log_pt.squeeze(-1)  # Shape: (batch_size)

#         # Compute the probabilities for the correct class
#         pt = pt.gather(dim=-1, index=target.unsqueeze(-1).to(device))  # Shape: (batch_size, 1)
#         pt = pt.squeeze(-1)  # Shape: (batch_size)

#         # Compute the cross-entropy loss for the correct class
#         cross_entropy_loss = -log_pt  # Shape: (batch_size)

#         # Focal loss scaling factor
#         alpha_t = self.alpha[target] if isinstance(self.alpha, torch.Tensor) else self.alpha
#         # alpha_t = alpha_t.to(device)  # Move to the same device

#         # Focal loss formula
#         loss = alpha_t * ((1 - pt) ** self.gamma) * cross_entropy_loss  # Shape: (batch_size)
#         loss = loss.to(device)

#         # Apply class weights if provided
#         if self.weight is not None:
#             loss = loss * self.weight.gather(dim=0, index=target).to(device)  # Ensure weights and target are on the same device

#         # Reduction (mean or sum)
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // 16, bias=False)
        self.fc2 = nn.Linear(channels // 16, channels, bias=False)

    def forward(self, x):
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(-1)))))
        return x * scale.unsqueeze(-1)

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

# LightningModule that handles the model and training/validation loop
class CustomCNN(pl.LightningModule):
    def __init__(self, num_classes, num_slices, dataset, alpha, gamma, margin, emb_dim):
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

        # # Updated Convolutional Layers with num_slices as input channels
        self.conv1 = nn.Conv1d(in_channels=num_slices, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        # self.conv4 = nn.Conv1d(num_slices * 4, num_slices * 8, kernel_size=5, stride=1, padding=2)
        # self.conv5 = nn.Conv1d(num_slices * 16, num_slices * 32, kernel_size=4, stride=1, padding=2)

        self.gn1 = nn.GroupNorm(2, 16, dtype=torch.bfloat16)
        self.gn2 = nn.GroupNorm(4, 32, dtype=torch.bfloat16)
        self.gn3 = nn.GroupNorm(8, 64, dtype=torch.bfloat16)
        self.gn4 = nn.GroupNorm(16, 128, dtype=torch.bfloat16)

        # Convert layers to bfloat16 for performance
        self.conv1 = self.conv1.to(torch.bfloat16)
        self.conv2 = self.conv2.to(torch.bfloat16)
        self.conv3 = self.conv3.to(torch.bfloat16)
        self.conv4 = self.conv4.to(torch.bfloat16)
        # self.conv5 = self.conv5.to(torch.bfloat16)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(8)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 8, emb_dim, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(emb_dim, num_classes, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(p=0.4)

        # Losses
        self.classification_loss = FocalLoss(alpha=alpha, gamma=gamma, weight=self.class_weights)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)


    def forward(self, x, embeddings=False):
        batch_size, num_slices, _, _ = x.size()
        # Reshape input to match the new channel size for Conv1D
        x = x.view(batch_size, num_slices, -1)  # Shape: (batch_size, num_slices, num_slices * num_slices)
        

        # Apply Conv1D layers
        conv_output = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.01)
        conv_output = F.leaky_relu(self.gn2(self.conv2(conv_output)), negative_slope=0.01)
        conv_output = F.leaky_relu(self.gn3(self.conv3(conv_output)), negative_slope=0.01)
        conv_output = F.leaky_relu(self.gn4(self.conv4(conv_output)), negative_slope=0.01)
        # conv_output = F.leaky_relu(self.conv5(conv_output), negative_slope=0.01)

        # Global Pooling
        pooled_output = self.global_pool(conv_output)

        # Flatten the output
        flattened_output = pooled_output.view(batch_size, -1)

        # Fully Connected layers with dropout
        embedding = F.leaky_relu(self.fc1(flattened_output), negative_slope=0.01)
        dropout = self.dropout(embedding)

        # Output layer
        output = self.fc2(dropout)

        return output, embedding
    
    
    def steps(self, anchor, positive, negative, type, batch_size):
        anchor_input, anchor_label, target = anchor
        positive_input, _ = positive
        negative_input, _= negative

        ## Forward pass
        anchor_output, anchor_embedding = self(anchor_input)         # Anchor is the sample being trained on
        _, positive_embedding = self(positive_input)     # Positive is a sample in the same class
        _, negative_embedding = self(negative_input)     # Negative is a sample in a different class
        

        ## Loss Functions
        loss_classification = self.classification_loss(anchor_output, anchor_label)
        loss_triplet = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss = loss_classification + 0.5 * loss_triplet  # Weighting losses

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
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "train", self.trainer.datamodule.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "val", self.trainer.datamodule.batch_size)

        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "test", self.trainer.datamodule.batch_size)

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
            plt.savefig("tb_logs/pointnet_test/version_" + str(self.logger.version) + "/" +  name + "_confusion_matrix.png")
            plt.close(fig)

            # Log the confusion matrix image to TensorBoard (uncomment if needed)
            # self.logger.experiment.add_image("Confusion Matrix", 
            #                                   self.tensorboard_image_from_figure(fig), 
            #                                   self.current_epoch)

    
    def log_embeddings(self, labels, embeddings, class_names):
        with torch.no_grad():
            embeddings =  embeddings.to(torch.float32)
            # Convert numerical labels to string class names
            string_labels = [class_names[label.item()] for label in labels]
            
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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-4)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3, cooldown=2, min_lr=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Loss/val_loss'}