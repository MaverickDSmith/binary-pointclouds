import argparse
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.cli import LightningCLI
from lightning.pytorch.profilers import PyTorchProfiler
# from models.Conv1dCNN import CustomCNN
from models.PointNet import PointNet
from models.miner_test import CustomCNN
from models.t_net_test import CustomCNN
# from models.voxnet_testing import CustomCNN
# from models.xyz_test import CustomCNN
# from dataloaders.BitArrayDataset import PointCloudDataModule
# from dataloaders.BA_Dataset_split import PointCloudDataModule
# from dataloaders.xyz_dataloader import PointCloudDataModule
# from dataloaders.miner_metric import PointCloudDataModule
from dataloaders.realtime_encoding import PointCloudDataModule


# TODO:
# 1.) Update base config template
# 2.) Determine a profiler that I actually like
# 3.) Add ability to load different model and dataloader types
# 4.) Everything should be modifiable via config, either add it or make it clear what is hardcoded
# 5.) Clean up imports (you should be able to do some stuff with the __init__.py files to make this cleaner)

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
    # TODO: Determine if this should be set dynamically or via config
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

    # Train the model
    # TODO:
    # Figure out what to do with log_every_n_steps
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with PyTorch Lightning.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')

    args = parser.parse_args()
    train_main(config_path=args.config)

