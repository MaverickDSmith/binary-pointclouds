import argparse
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.cli import LightningCLI
from lightning.pytorch.profilers import AdvancedProfiler
# from models.Conv1dCNN import CustomCNN
from models.PointNet import PointNet
# from models.miner_test import CustomCNN
from models.xyz_test import CustomCNN
# from dataloaders.BitArrayDataset import PointCloudDataModule
# from dataloaders.BA_Dataset_split import PointCloudDataModule
from dataloaders.xyz_dataloader import PointCloudDataModule
# from dataloaders.miner_metric import PointCloudDataModule

def cli_main():
    cli = LightningCLI(CustomCNN, PointCloudDataModule)

def train_main(config_path='config.yaml'):
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize DataModule
    datamodule = PointCloudDataModule(
        root_dir=config['data']['root_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    datamodule.setup()

    # Determine number of slices from the dataset
    # _, _, _, num_slices = next(iter(datamodule.train_dataloader()))
    # _, _, num_slices = next(iter(datamodule.train_dataloader()))
    # num_slices = num_slices[0].item()
    num_slices = 65

    # Initialize Model
    if config['training']['checkpoint']:
        model = CustomCNN.load_from_checkpoint(config['training']['checkpoint_path'],
                                                num_classes=config['model']['num_classes'],
                                                num_slices=num_slices,
                                                alpha=config['model']['alpha'],
                                                gamma=config['model']['gamma'],
                                                margin=config['model']['margin'],
                                                emb_dim=config['model']['emb_dim'],
                                                lr = config['model']['learning_rate'])
    else:
        model = CustomCNN(num_classes=config['model']['num_classes'],
                          num_slices=num_slices,
                          alpha=config['model']['alpha'],
                          gamma=config['model']['gamma'],
                          margin=config['model']['margin'],
                          emb_dim=config['model']['emb_dim'],
                          dataset=datamodule,
                          lr = config['model']['learning_rate'])

    # Logging and Checkpointing
    logger = TensorBoardLogger(config['logging']['log_dir'], name=config['logging']['log_name'])
    checkpoint_callback = ModelCheckpoint(
        monitor=config['callbacks']['checkpoint']['monitor'],
        save_top_k=config['callbacks']['checkpoint']['save_top_k'],
        mode=config['callbacks']['checkpoint']['mode'],
        filename=config['callbacks']['checkpoint']['filename']
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train the model with profiler
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if config['gpu']['use_gpu'] else 'cpu',
        devices=config['gpu']['devices'],
        strategy=config['gpu']['strategy'],
        num_nodes=config['gpu']['num_nodes'],
        log_every_n_steps=30
    )
    trainer.fit(model, datamodule)

    # Test the model
    trainer.test(model, datamodule.test_dataloader())

    # Save the model if specified
    if config['training']['save_model']:
        model_path = "final_model.ckpt"
        trainer.save_checkpoint(model_path)
        print(f"Model saved at {model_path}")

def test_main():
    # Step 1: Load the previously saved model
    model_path = "final_model.ckpt"  # Path to your saved model checkpoint
    loaded_model = CustomCNN.load_from_checkpoint(model_path, num_classes=40, num_slices=65)

    # Step 2: Set up the DataModule (same as before)
    root_dir = '/home/hi5lab/pointcloud_data/dataset/slice_64_voxel_rle'
    datamodule = PointCloudDataModule(root_dir)
    datamodule.setup()
    logger = TensorBoardLogger("tb_logs", name="pointcloud_cnn_voxel_rle")

    # Step 3: Create a Trainer instance without training
    trainer = Trainer(
        logger=logger,  # You can reuse the logger if needed
        accelerator='gpu',  # Ensure you're using GPUs
        devices=1,  # Set to the number of GPUs you want to use
        strategy=DDPStrategy(find_unused_parameters=False),  # Enable DDP if using multiple GPUs
        num_nodes=1  # This will be more than 1 when scaling to multiple GPUs
    )

    # Step 4: Run the test step
    trainer.test(loaded_model, datamodule.test_dataloader())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with PyTorch Lightning.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')

    args = parser.parse_args()
    train_main(config_path=args.config)
    # cli_main()  # Uncomment if you want to use CLI
    # test_main()  # Uncomment if you want to run testing
