#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import random
from logging import Logger

# Third-party libraries
import torch
import deepspeed
import numpy as np
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

# User define module
from hammer.utils.config import Config
from hammer.algorithms.uitls import mpu
from hammer.algorithms.trainer.checkpoint import load_checkpoint
from hammer.algorithms.trainer.average_meter import AveragedMeter
from hammer.algorithms.models.model_factory import ModelFactory
from hammer.algorithms.dataset.dataset_factory import DatasetFactory


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Trainer(object):

    def __init__(self, model_factory: ModelFactory, dataset_factory: DatasetFactory, config: Config, logger: Logger):
        """_summary_

        Args:
            model_factory (ModelFactory): _description_
            dataset_factory (DatasetFactory): _description_
            config (Config): _description_
            logger (Logger): _description_
        """
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory
        self.config = config
        self.logger = logger

    def _init_distributed(self, config):
        
        # 
        device = config.train.rank % torch.cuda.device_count()
        # 
        if config.train.local_rank is not None:
            device = config.local_rank
        torch.cuda.device(device)
        # deepspeed 
        if config.train.use_deepspeed:
            deepspeed.init_distributed()
        else:
            master_addr = config.train.master_addr
            master_port = config.train.master_port
            init_method = f'tcp://{master_addr}:{master_port}'
            # 
            distributed.init_process_group(
                backend=config.train.distributed_backend,
                world_size=config.train.world_size,
                rank=config.train.rank,
                init_method=init_method
            )
        # 
        if config.train.use_deepspeed and self.train.deepspped.activation_checkpointing.enabled:
            #
            mpu.initialize_model_parallel(config.train.model_parallel_size)
            #
            deepspeed.checkpointing.configure(
                mpu=mpu,
                deepspeed_config=config.train.deepspeed_config,
                num_checkpoints=config.model.num_layers
            )
            #
            mpu.checkpoint = deepspeed.checkpointing.checkpoint
            mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed   

    def set_random_seed(self, seed: int):
        """Set random seed for training.

        Args:
            seed (int): seed for training.
        """
        # invalid seed
        if seed is None and seed < 0:
            return
        # set random seed 
        random.seed(seed)  
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

    def train_step(self, model, optimizer, data_loader):

        model.train()
        # reset gradient if use_deepspeed
        if self.config.train.use_deepspeed:
            optimizer.zero_grad()

        for data, target in data_loader:
            # forward pass
            output = model(data)

    def train(self):
        """_summary_
        """
        # initialize distributed training
        self._init_distributed(self.config)
        # set random seed
        self.set_random_seed(self.config.train.seed)
        # create model
        model = self.model_factory.create_model()
        # create optimizer for training
        optimizer = self.model_factory.create_optimizer()
        # create learning rate scheduler for training
        lr_scheduler = self.model_factory.create_lr_scheduler()
        # deepspeed 
        if self.config.train.use_deepspeed:
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=self.config,
                lr_scheduler=lr_scheduler,
                model_parameters=model.parameters(),
                mpu=mpu,
                dist_init_required=False
            )
        
        iteration = 0
        # create dataloader for training
        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset.get_train(),
            batch_size=self.train.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=2,
            collate_fn=None,
            pin_memory=True,
            drop_last=False
        )
        # create dataloader for validation
        if self.config.valid.enabled:
            valid_dataloader = torch.utils.data.DataLoader(
                dataset=self.dataset.get_valid(),
                batch_size=self.valid.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=2,
                collate_fn=None,
                pin_memory=True,
                drop_last=False
            )
        # resume training from given checkpoint
        if self.config.checkpoint_path is not None:
            iteration = load_checkpoint(model, optimizer, lr_scheduler, self.config)
        # train model
        while iteration < self.config.train.max_iteration:
            # train model for one iteration
            self.train_step(model, train_dataloader)
            # evaluate model and print results
            if self.config.valid.enabled:
                self.evaluate_and_print_results('valid', model, valid_dataloader)
            iteration += 1

    def evaluate(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> dict:
        """This method evaluates the model by running it on the given data loader.

        Args:
            model (torch.nn.Module): the model to be evaluated.
            data_loader (torch.utils.data.DataLoader): the data loader for the evaluation data.

        Returns:
            dict: the results of the evaluation.
        """
        metrics = {}
        # turn on evaluation mode 
        model.eval()
        with torch.no_grad():
            # evaluate model on given data loader
            for data, target in data_loader:
                # forward pass
                output = model(data)
                # calculate metrics
                batch_metrics = model.metrics(output, target)
                # update metrics
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = AveragedMeter(key)
                    metrics[key].update(value)
        return {key: metrics[key].average() for key in metrics}

    def evaluate_and_print_results(self, prefix: str, model: any, data_loader: torch.utils.data.DataLoader, global_step: int, summary_writer: SummaryWriter):
        """Evaluate the model on the given data loader and print the results.
        """
        # evaluate the model on the gven data loader
        metrics = self.evaluate(model, data_loader, self.config)
        # add evalation metrics to summary 
        if summary_writer is not None:
            for key, meter in metrics.items():
                summary_writer.add_scalar(f'{prefix}/{key}', meter.average, global_step)
        # print the evalation results
        for key, meter in metrics.items():
            self.logger.info(f'{prefix}/{key}: {meter.average}')
