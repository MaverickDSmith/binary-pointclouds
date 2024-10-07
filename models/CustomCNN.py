import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# LightningModule that handles the model and training/validation loop
class CustomCNN(pl.LightningModule):
    def __init__(self, num_classes, num_slices):
        super(CustomCNN, self).__init__()
        self.num_slices = num_slices
        self.num_classes = num_classes
        self.val_preds = []  
        self.val_labels = []  
        self.test_preds = []  
        self.test_embeddings = [] 
        self.test_labels = []  

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)

        # Convert convolutional layers to bfloat16 to match input tensor
        self.conv1 = self.conv1.to(torch.bfloat16)
        self.conv2 = self.conv2.to(torch.bfloat16)
        self.conv3 = self.conv3.to(torch.bfloat16)

        # Global Pool to learn features from full set of slices
        self.global_pool = nn.AdaptiveMaxPool2d((4, 4))

        # Regularizations
        self.fc1 = nn.Linear(16 * num_slices * 4 * 4, 256, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(256, num_classes, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(p=0.25)

        # Losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.5)

    def forward(self, x, embeddings=False):
        batch_size, num_slices, _, _ = x.size()
        outputs = []
        
        # Iterate through each slice, sending each slice through the Conv2D Layers
        # Add their output to a list
        for slice_idx in range(num_slices):
            slice_input = x[:, slice_idx:slice_idx+1, :, :]
            conv_output = F.leaky_relu(self.conv1(slice_input), negative_slope=0.01)
            conv_output = F.leaky_relu(self.conv2(conv_output), negative_slope=0.01)
            conv_output = F.leaky_relu(self.conv3(conv_output), negative_slope=0.01)

            outputs.append(conv_output)

        # Concatenate each slice's features into a single tensor, then do a global_pooling pass
        fused_output = torch.cat(outputs, dim=1)
        fused_output = self.global_pool(fused_output)
        flattened_output = fused_output.view(batch_size, -1)

        # If we want only the embeddings, output here before final layer
        if embeddings:
            return F.leaky_relu(self.fc1(flattened_output), negative_slope=0.01)

        # Regularization, then output
        fc_output = self.dropout(F.leaky_relu(self.fc1(flattened_output), negative_slope=0.01))
        output = self.fc2(fc_output)
        return output
    
    def steps(self, anchor, positive, negative, type):
        anchor_input, anchor_label = anchor
        positive_input, _ = positive
        negative_input, _= negative

        ## Forward pass
        anchor_output = self(anchor_input)         # Anchor is the sample being trained on
        positive_output = self(positive_input)     # Positive is a sample in the same class
        negative_output = self(negative_input)     # Negative is a sample in a different class
        
        ## Loss Functions
        loss_classification = self.classification_loss(anchor_output, anchor_label)
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        loss = loss_classification + 0.6 * loss_triplet  # Weighting losses

        ## Accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        ## Logging
        self.steps_log(loss, loss_classification, loss_triplet, acc, preds, type)

        ## Storing for future evals
        if type == "val":
            self.val_preds.append(preds)
            self.val_labels.append(anchor_label)

        if type == "test":
            embedding = self(anchor_input, True)
            self.test_preds.append(preds)
            self.test_labels.append(anchor_label)
            self.test_embeddings.append(embedding)

        return loss

    def steps_log(self, loss, loss_classification, loss_triplet, acc, preds, type):
        self.log(f'Loss/{type}_loss', loss, prog_bar=True)
        self.log(f'Loss/{type}_class_loss', loss_classification, prog_bar=False)
        self.log(f'Loss/{type}_trip_loss', loss_triplet, prog_bar=False)
        self.log(f'Acc/{type}_acc', acc, prog_bar=True)


    def training_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "val")

        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_tensor, positive_tensor, negative_tensor, _ = batch
        loss = self.steps(anchor_tensor, positive_tensor, negative_tensor, "test")

        return loss

    def on_val_epoch_end(self):
        # Compute confusion matrix
        all_preds = torch.cat(self.val_preds)
        all_labels = torch.cat(self.val_labels)

        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(self.num_classes))
        self.log_confusion_matrix(cm, "Validation")

        # Clear the stored predictions and labels
        self.val_preds.clear()
        self.val_labels.clear()

    def on_test_epoch_end(self):
        # Compute confusion matrix
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        all_embeddings = torch.cat(self.test_embeddings)

        # Ensure consistent length
        assert len(all_labels) == len(all_preds), "Length mismatch between labels and predictions."

        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(self.num_classes))
        self.log_confusion_matrix(cm, "Test")

        self.log_embeddings(all_labels, all_embeddings)

        # Clear the stored test embeddings and labels
        self.test_preds.clear()
        self.test_embeddings.clear()
        self.test_labels.clear()
    
    def log_confusion_matrix(self, cm, name):
        # Plot the confusion matrix and log it to TensorBoard
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        plt.title(name +' Confusion Matrix')
        plt.tight_layout()

        # Save the figure
        plt.savefig(name + "_confusion_matrix.png")
        plt.close(fig)

        # Log the confusion matrix image to TensorBoard
        # self.logger.experiment.add_image("Confusion Matrix", 
        #                                   self.tensorboard_image_from_figure(fig), 
        #                                   self.current_epoch)
    
    def log_embeddings(self, labels, embeddings):
        # Log the embeddings to TensorBoard
        self.logger.experiment.add_embedding(embeddings, metadata=labels, global_step=self.current_epoch)

    @staticmethod
    def tensorboard_image_from_figure(fig):
        """Convert a matplotlib figure to a tensorboard image."""
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Loss/val_loss'}