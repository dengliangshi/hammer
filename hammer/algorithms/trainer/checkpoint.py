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

# User define module
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


def unwrap_model(model, module_instances=None):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_checkpoint_name(checkpoint_path, iteration, ):
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoint_path, directory,
                            'mp_rank_{:02d}'.format(
                                mpu.get_tensor_model_parallel_rank()),
                            'model_optim_rng.pt')
    return os.path.join(checkpoint_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            mpu.get_tensor_model_parallel_rank(),
                            mpu.get_pipeline_model_parallel_rank()),
                        'model_optim_rng.pt')


def save_ds_checkpoint(iteration, model, lr_scheduler, config):
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


def save_checkpoint(iteration, model, lr_scheduler, config):

    if config.train.use_deepspeed:
        save_ds_checkpoint(iteration, model, lr_scheduler, )
    else:

        if config.train.use_deepspeed:
            model = model.module


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, config):
    """Save a model checkpoint."""

    # only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, config.train.checkpoint_path))

    # collect rng state across data parallel ranks
    rng_state = get_rng_state()

    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['config'] = config
        state_dict['checkpoint_version'] = 1.0
        state_dict['iteration'] = iteration
        if len(model) == 1:
            state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        # Save.
        checkpoint_name = get_checkpoint_name(args.save, iteration)
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
