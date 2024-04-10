#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json
import random
from logging import Logger

# Third-party libraries
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# User define module
from hammer.utils.config import Config
from hammer.algorithms.uitls import mpu


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
def print_rank_0(message: str):
    """If distributed is initialized, print only on rank 0.

    Args:
        message (str): message to display.
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_checkpoint_name(checkpoint_path: str, iteration: int):
    """Get the checkpoint file name.

    Args:
        checkpoint_path (str): the path to save the checkpoint.
        iteration (int): iteration number of the current training process.
        
    Returns:
        str: the checkpoint file name.
    """
    directory = 'iter_{:07d}'.format(iteration)
    # use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoint_path, directory,
            'mp_rank_{:02d}'.format(mpu.get_tensor_model_parallel_rank()), 'model.pt')
    return os.path.join(checkpoint_path, directory,
        'mp_rank_{:02d}_{:03d}'.format(mpu.get_tensor_model_parallel_rank(),
        mpu.get_pipeline_model_parallel_rank()), 'model.pt')


def save_ds_checkpoint(iteration: int, model: torch.Module, lr_scheduler: LRScheduler, config: Config):
    """Save checkpoint with DeepSpeed.

    Args:
        iteration (int): iteration number of the current training process.
        model (torch.Module): the instance of model to save.
        lr_scheduler (LRScheduler): learning rate scheduler.
        config (Config): user defined configuration.
    """
    state_dict = {}
    state_dict['iteration'] = iteration
    state_dict['consumed_train_samples'] = config.train.consumed_train_samples
    # save state of learning rate sheduler if given
    if lr_scheduler is not None:
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()
    # save random number generator states if enabled
    if config.train.save_rng_state:
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
        state_dict['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    model.save_checkpoint(config.train.checkpoint_path, iteration, client_state=state_dict)


def save_checkpoint(iteration: int, model: torch.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, config: Config):
    """Save a model checkpoint.

    Args:
        iteration (int): iteration number of the current training process.
        model (torch.Module): the instance of model to save.
        lr_scheduler (LRScheduler): learning rate scheduler.
        config (Config): user defined configuration.
    """
    meta_data = {'iteration': iteration}
    # save model checkpoint with deepspeed.
    if config.train.use_deepspeed and config.train.save_deepspeed_state:
        save_ds_checkpoint(iteration, model, lr_scheduler, config)
    # save checkpoint only on rank 0.
    elif mpu.get_data_parallel_rank() == 0:
        if config.train.use_deepspeed:
            model = model.module
        state_dict = {'iteration': iteration}
        state_dict['module'] = model.state_dict()
        # save the optimizer and lr_scheduler state dict if enabled.
        if config.train.save_optimizer_state:
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        # save the random number generator states if enabled.
        if config.train.save_rng_state:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
        torch.save(state_dict, get_checkpoint_name(config.train.checkpoint_path, iteration))
    # barrier to make sure all the processes have finished saving the checkpoint.
    torch.distributed.barrier()
    # save the tracker file.
    if mpu.get_data_parallel_rank() == 0:
        tracker_file = os.path.join(config.train.checkpoint_path, 'meta.json')
        with open(tracker_file, 'w', encoding='utf-8') as output_file:
            json.dump(meta_data, output_file, ensure_ascii=False, indent=4)
        print_rank_0('Saved checkpoint to {}'.format(tracker_file))


def load_checkpoint(model, optimizer, lr_scheduler, config):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        lr_scheduler (_type_): _description_
        config (_type_): _description_

    Returns:
        int: _description_
    """
    # load in meta data of checkpoint
    tracker_file = os.path.join(config.train.checkpoint_path, 'meta.json')
    meta_data = json.load(open(tracker_file, 'r', encoding='utf-8'))
    # load checkpoint with deepspeed.
    if config.train.use_deepspeed and config.train.load_deepspeed_state:
        checkpoint_name, state_dict = model.load_checkpoint(
            ckpt_dir=config.train.checkpoint_path,
            ckpt_id=meta_data['iteration'],
            load_optimizer_states=config.train.load_optimizer_state,
            load_lr_scheduler_states=config.train.load_lr_scheduler_states,
            load_module_only=True
        )
    else:
        checkpoint_name = get_checkpoint_name(config.train.checkpoint_path, iteration)
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    # 
    if config.train.use_deepspeed:
        model = model.module
    # load the model state dict
    model.load_state_dict(state_dict['module'])
    # load the optimizer state dict
    if optimizer is not None and 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    # load the learning rate scheduler state dict
    if lr_scheduler is not None and 'lr_scheduler' in state_dict:
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    # load the random number generator states if enabled.
    if config.train.load_rng_state:
        random.setstate(state_dict['random_rng_state'])
        np.random.set_state(state_dict['np_rng_state'])
        torch.set_rng_state(state_dict['torch_rng_state'])
        torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
        mpu.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])
    
    iteration = state_dict['iteration']

    return iteration
