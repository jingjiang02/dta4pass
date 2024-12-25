"""Filesystem utility functions."""
from __future__ import absolute_import

import errno
import glob
import logging
import os

import torch

from ..config import cfg


def save_checkpoint(args, model, epoch, optimizer=None, lr_scheduler=None, is_best=False,
                    best_save_name='best_model.pth', not_best_save_name=''):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.TRAIN.MODEL_SAVE_DIR)
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    filename = '{}{}_epoch_{}.pth'.format(cfg.TIME_STAMP, not_best_save_name, str(epoch))
    if is_best:
        best_filename = f'{cfg.TIME_STAMP}_{best_save_name}'
        best_filename = os.path.join(directory, best_filename)
        try:
            torch.save(model_state_dict, best_filename)
        except OSError:
            print(f"save ckpt [{best_filename}] error!")
            logging.error(f"save ckpt [{best_filename}] error!")
    else:
        pre_filename = glob.glob('{}/{}{}_epoch*.pth'.format(directory, cfg.TIME_STAMP, not_best_save_name))
        try:
            for p in pre_filename:
                os.remove(p)
        except OSError as e:
            logging.info(e)

        # save epoch
        save_state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
        }
        filename = os.path.join(directory, filename)
        if not args.distributed or (args.distributed and args.local_rank % args.num_gpus == 0):
            try:
                torch.save(save_state, filename)
            except OSError:
                print(f"save ckpt [{filename}] error!")
                logging.error(f"save ckpt [{filename}] error!")


def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
