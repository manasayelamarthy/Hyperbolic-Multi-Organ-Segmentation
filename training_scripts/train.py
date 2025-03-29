
import os
import sys
import time
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import *
from datasets import get_dataloaders
from models import model_trainers
from utils import *

from validation import Validator
from test import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ['train', 'validation', 'test'], help = 'choose between train, validation, test')
    parser.add_argument('--version', type = int, help = 'version of the config')
    parser.add_argument('--dataset', help = 'name of the dataset')

    parser.add_argument('--data-dir', type = str, help = 'path to the dataset dir')
    parser.add_argument('--all-configs-dir', type = str, help = 'path to the all configs dir')
    parser.add_argument('--split', type = str, choices = ['training', 'validation', 'inference'], help = "choose between 'training', 'validation', 'inference'")
    parser.add_argument('--image-size', type = int, nargs = 2, help = 'dimensions of image') 
    parser.add_argument('--labels', type = str, nargs = '*', help = 'list of labels to be segmented')
    parser.add_argument('--window', type = int, nargs = 2, help = 'windowing of image')
    parser.add_argument('--window-preset', choices = ['ct_abdomen','ct_liver','ct_spleen','ct_pancreas'], help = 'choose between window presets')
    parser.add_argument('--transform', action = 'store_true', help = 'apply transformations to the image')

    parser.add_argument('--model', choices = ['unet', 'hc_unet'], help = 'choose between unet and hc_unet')
    parser.add_argument('--loss', help = 'loss function to be used',
                        choices=['dice', 'cross_entropy', 'jaccard', 'hyperul', 'hyperbolicdistance'])
    parser.add_argument('--loss-list', nargs = '*', help = 'list of loss functions to be used',
                        choices=['dice', 'cross_entropy', 'jaccard', 'hyperul', 'hyperbolicdistance'])
    parser.add_argument('--weights', nargs = '*', type = float, help = 'list of weights for loss functions')
    parser.add_argument('--lr', type = float, help = 'learning rate')

    parser.add_argument('--metric', choices = ['all', 'miou', 'precision', 'recall', 'dice'], help = 'choose between metrics')
    parser.add_argument('--batch-size', type = int, help = 'batch size')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--checkpoint-dir', type = str, help = 'path to the checkpoint dir')

    parser.add_argument('--single-gpu', action= 'store_true', help= 'use single gpu for training')
    parser.add_argument('--visualize', action = 'store_true', help = 'save visualizations of the training logs & random sample inference')

    return parser.parse_args()


class Trainer:
    def __init__(self, train_data : DataLoader,
                 trainer, epochs: int,
                 validator, val_data : DataLoader,
                 criterion, metrics,
                 multi_gpu: bool,
                 train_logger,
                 config_filename,
                 train_config):
        
        self.config = train_config
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if multi_gpu else 0
        
        self.data = train_data
        self.model : nn.Module = trainer.model.to(self.gpu_id)
        self.criterion = criterion
        self.metrics = metrics
        self.optimizers = trainer.optimizers
        self.epochs = epochs

        self.multi_gpu = multi_gpu
        if multi_gpu:
            self.model = DDP(self.model, device_ids = [self.gpu_id])

        self.validator = validator(
            val_data = val_data,
            criterion = criterion,
            metrics = metrics,
            multi_gpu = multi_gpu
        )
        self.best_val_dice = 0
        self.logger = train_logger
        self.filename = config_filename
    
    def _run_epoch(self, epoch : int) -> dict:

        # Initialize epoch logs to 0
        epoch_logs : dict = {
            'loss' : 0,
            **{metric.name: 0 for metric in self.metrics}
        }

        train_iterator = tqdm(self.data, total = len(self.data), desc = f"Epoch {epoch + 1}")
        # Train on all Batches
        for inputs, masks in train_iterator:
            inputs = inputs.to(self.gpu_id)
            masks = masks.to(self.gpu_id)

            loss, metrics = self._run_batch(inputs, masks)

            # Accumulate Logs
            epoch_logs['loss'] += loss
            for metric in metrics:
                epoch_logs[metric] += metrics[metric]

        # Compute Average for all logs 
        for log in epoch_logs:
            epoch_logs[log] /= len(self.data)
        
        return epoch_logs

    def _run_batch(self, inputs, masks):

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, masks)

        loss.backward()

        for optimizer in self.optimizers:
            optimizer.step()

        outputs = torch.argmax(outputs, dim = 1)

        metrics = {metric.name: 0 for metric in self.metrics}
        for metric in self.metrics:
            metrics[metric.name] += metric.compute(outputs, masks.squeeze(1))[1]
            
        return loss.item(), metrics
            
    def train(self):
        start_time = time.time()
        self.model.train()

        if self.gpu_id == 0:
            print(f"Training {self.config.model} for {self.epochs} epochs on {torch.cuda.device_count()} GPUs")
        for epoch in range(self.epochs):
            epoch_logs = self._run_epoch(epoch)

            # Validation
            val_logs = self.validator.validate(model = self.model)

            # Update Logs
            self.logger.add_epoch_logs(epoch, epoch_logs, val_logs)

            # Save if best model checkpoint
            if val_logs['dice_score'] > self.best_val_dice:
                save_checkpoint(self.model, self.optimizers, epoch,
                                self.config.checkpoint_dir + '/best_model.pth',
                                self.multi_gpu)
        # Save Logs
        self.logger.save_train_logs(filename = self.config.checkpoint_dir + '/train_logs.csv')

        training_time = time.time() - start_time
            
        if self.gpu_id == 0:
            print(f"Training Complete in {training_time:.2f}s with {training_time/self.epochs:.2f} for epcoh")
            print(f"Best Validation Dice Score: {self.best_val_dice:.4f}")
        

def main():
    init_process_group(backend='nccl')
    gpu_id = os.environ['LOCAL_RANK']
    args = parse_args().__dict__

    if not args['data_dir']:
        raise Exception("Data Directory is not provided")

    if not torch.cuda.is_available():
        raise Exception("Cuda is not available, training on CPU is not ideal")
    
    multi_gpu = not args['single_gpu'] and torch.cuda.device_count() > 1

    all_config = allConfig(**args)

    config_filename = all_config.get_config_filename()
    all_config.save_config(all_config.all_configs_dir + config_filename)

    checkpoint_dir = os.path.join(all_config.train_config['checkpoint_dir'], config_filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = file_logging(checkpoint_dir)

    if gpu_id == 0:
        logger.debug(f"Initialized process group with {torch.cuda.device_count() if multi_gpu else 1} GPUs.")
        
        logger.debug(f"Saved config details to {checkpoint_dir + '/config.json'}")
    
    all_config.save_config(os.path.join(checkpoint_dir, 'config'))
    
    train_config = trainConfig(**args)
    train_config.checkpoint_dir = checkpoint_dir
    
    
    trainDataloader, valDataloader = get_dataloaders(multi_gpu=multi_gpu, config=amosDatasetConfig(**args))
    if gpu_id == 0:
        logger.debug("Loaded train and validation dataloaders")
    
    labels = trainDataloader.dataset.labels
    labels_to_pixels = trainDataloader.dataset.label_to_pixel_value
    
    model_trainer = model_trainers[train_config.model]()
    if gpu_id == 0:
        logger.debug("Initialized model trainer")
    
    if args['loss_list'] is not None:
        criterion = criterions['combined'](
            loss_list=train_config.loss_list,
            weights=train_config.weights,
            labels=labels,
            labels_to_pixels=labels_to_pixels
        )
        train_config.loss = 'combined'
    else:
        criterion = criterions[train_config.loss](labels=labels, labels_to_pixels=labels_to_pixels)
    
    if train_config.metric == 'all':
        metrics = [metric(labels=labels, labels_to_pixels=labels_to_pixels) for metric in all_metrics.values()]
    else:
        metrics = [all_metrics[train_config.metric](labels=labels, labels_to_pixels=labels_to_pixels)]
    if gpu_id == 0:
        logger.debug("Initialized Loss and Metrics")
    
    train_logger = trainLogging(metrics=[metric.name for metric in metrics])
    if gpu_id == 0:
        logger.debug("Initialized training logger")
    
    trainer = Trainer(
        train_data=trainDataloader,
        validator=Validator,
        val_data=valDataloader,
        trainer=model_trainer,
        epochs=train_config.epochs,
        criterion=criterion,
        metrics=metrics,
        train_logger=train_logger,
        config_filename=config_filename,
        multi_gpu=multi_gpu,
        train_config=train_config
    )
    if gpu_id == 0:
        logger.debug("Initialized trainer")
    
    trainer.train()
    if gpu_id == 0:
        logger.debug("Training completed")
    
    if args['visualize']:
        if gpu_id == 0:
            logger.debug("Starting visualization")
        tester = Tester(
            test_data=valDataloader.dataset,
            trainer=trainer,
            criterion=criterion,
            metrics=metrics,
            checkpoint_path=os.path.join(train_config.checkpoint_dir, 'best_model.pth'),
            n_samples=32,
            batch_size=args['batch_size'],
            random_seed=42
        )
        _, images, masks, preds = tester.infer()
        
        os.makedirs(os.path.join(train_config.checkpoint_dir, 'visualizations'), exist_ok=True)
        log_visualizer = trainLogVisualizer(os.path.join(train_config.checkpoint_dir, 'train_logs.csv'))
        log_visualizer.visualize(save_path=os.path.join(train_config.checkpoint_dir, 'visualizations/logs.png'))
        
        if gpu_id == 0:
            logger.debug("Training log visualizations are saved.")

        infer_visualizer = inferVisualizer(criterion=criterion)
        infer_visualizer.visualize_batch(images, masks, preds, save_path=os.path.join(train_config.checkpoint_dir, 'visualizations/infer.png'))
        
        if gpu_id == 0:
            logger.debug("Inference Visualizations are saved.")
    destroy_process_group()

    
if __name__ == "__main__":
    main()
