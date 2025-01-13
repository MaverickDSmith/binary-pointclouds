import torch
from torch.utils.data import DataLoader
from dataloaders.DiffArrayDataset import BitArrayDataset
import pytorch_lightning as pl
from difflogic import LogicLayer, GroupSum

import argparse
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.cli import LightningCLI

# Dataset and DataLoader setup
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=64, num_workers=16):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BitArrayDataset(self.train_dir, "train")
        self.val_dataset = BitArrayDataset(self.val_dir, "val")
        self.test_dataset = BitArrayDataset(self.test_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=4)


# Define the model
class CustomLogicModel(pl.LightningModule):
    def __init__(self, device):
        super(CustomLogicModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(274625, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 512_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(512_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 1_024_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(1_024_000, 2_048_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(2_048_000, 2_048_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(2_048_000, 2_048_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            LogicLayer(2_048_000, 2_048_000, device=device, implementation='cuda', grad_factor=3, connections='unique'),
            GroupSum(k=40, tau=120)
        )
        self.class_criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    # def training_step(self, batch, batch_idx):
    #     anchor_input, anchor_label, _ = batch[0]
    #     outputs = self(anchor_input)
    #     loss = self.class_criterion(outputs, anchor_label)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     anchor_input, anchor_label, _ = batch[0]
    #     outputs = self(anchor_input)
    #     loss = self.class_criterion(outputs, anchor_label)
    #     # Accuracy calculation
    #     _, predicted = torch.max(outputs, 1)
    #     correct = (predicted == anchor_label).sum().item()
    #     total = anchor_label.size(0)
    #     accuracy = correct / total * 100
        
    #     # Save the outputs for later use in epoch end
    #     self.log('Loss/val_loss', loss, prog_bar=True)
    #     self.log('Acc/val_acc', accuracy, prog_bar=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     anchor_input, anchor_label, _ = batch[0]
    #     outputs = self(anchor_input)
    #     loss = self.class_criterion(outputs, anchor_label)
    #     # Accuracy calculation
    #     _, predicted = torch.max(outputs, 1)
    #     correct = (predicted == anchor_label).sum().item()
    #     total = anchor_label.size(0)
    #     accuracy = correct / total * 100
        
    #     # Save the outputs for later use in epoch end
    #     self.log('Loss/test_loss', loss, prog_bar=True)
    #     self.log('Acc/test_acc', accuracy, prog_bar=True)

    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
        
    def steps(self, anchor, type, batch_size):
        anchor_input, anchor_label = anchor


        ## Forward pass
        anchor_output = self(anchor_input)         # Anchor is the sample being trained on

        ## Loss Functions
        loss_classification = self.class_criterion(anchor_output, anchor_label)
        loss = loss_classification  

        ## Accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        ## Logging
        self.steps_log(loss, loss_classification, acc, preds, type, batch_size)

        return loss

    def steps_log(self, loss, loss_classification, acc, preds, type, batch_size):
        if type == "train":
            self.log(f'Loss/{type}_loss', loss, prog_bar=True, batch_size=batch_size)
            self.log(f'Loss/{type}_class_loss', loss_classification, prog_bar=False, batch_size=batch_size)
            self.log(f'Acc/{type}_acc', acc, prog_bar=True, batch_size=batch_size)
        else:
            self.log(f'Loss/{type}_loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f'Loss/{type}_class_loss', loss_classification, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log(f'Acc/{type}_acc', acc, prog_bar=True, batch_size=batch_size, sync_dist=True)


    def training_step(self, batch, batch_idx):
        anchor_tensor, _ = batch
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



def cli_main():
    cli = LightningCLI(CustomLogicModel, PointCloudDataModule)

def train_main(config_path='config.yaml'):
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice64"
    val_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice64"
    test_dir = "/home/hi5lab/pointcloud_data/storage_test_two/slice64"

    datamodule = PointCloudDataModule(train_dir, val_dir, test_dir)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomLogicModel(device)

    # Logging and Checkpointing
    logger = TensorBoardLogger(config['logging']['log_dir'], name=config['logging']['log_name'])
    checkpoint_callback = ModelCheckpoint(
        monitor=config['callbacks']['checkpoint']['monitor'],
        save_top_k=config['callbacks']['checkpoint']['save_top_k'],
        mode=config['callbacks']['checkpoint']['mode'],
        filename=config['callbacks']['checkpoint']['filename']
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train the model
    trainer = Trainer(
        max_epochs=200,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if config['gpu']['use_gpu'] else 'cpu',
        devices=config['gpu']['devices'],
        strategy=config['gpu']['strategy'],
        num_nodes=config['gpu']['num_nodes']
    )
    trainer.fit(model, datamodule)

    # Test the model
    trainer.test(model, datamodule.test_dataloader())

    # Save the model if specified
    if config['training']['save_model']:
        model_path = "final_model.ckpt"
        trainer.save_checkpoint(model_path)
        print(f"Model saved at {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with PyTorch Lightning.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')

    args = parser.parse_args()
    train_main(config_path=args.config)
    # cli_main()  # Uncomment if you want to use CLI
    # test_main()  # Uncomment if you want to run testing