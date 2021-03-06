""" Command-Line Interface Helpers """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import copy
import os
import warnings

import torch
from hydra import utils
from torch.utils.hipify.hipify_python import bcolors

from lightly.models import ZOO as model_zoo


def _custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return f"{bcolors.WARNING}{msg}{bcolors.WARNING}\n"


def print_as_warning(message: str):
    old_format = copy.copy(warnings.formatwarning)

    warnings.formatwarning = _custom_formatwarning
    warnings.warn(message, UserWarning)

    warnings.formatwarning = old_format


def cpu_count():
    """Returns the number of CPUs which are present in the system.

    This number is not equivalent to the number of available CPUs to the process.

    """
    return os.cpu_count()


def fix_input_path(path):
    """Fix broken relative paths.

    """
    if not os.path.isabs(path):
        path = utils.to_absolute_path(path)
    return path


def is_url(checkpoint):
    """Check whether the checkpoint is a url or not.

    """
    is_url = ('https://storage.googleapis.com' in checkpoint)
    return is_url


def get_ptmodel_from_config(model):
    """Get a pre-trained model from the lightly model zoo.

    """
    key = model['name']
    key += '/simclr'
    key += '/d' + str(model['num_ftrs'])
    key += '/w' + str(float(model['width']))

    if key in model_zoo.keys():
        return model_zoo[key], key
    else:
        return '', key


def load_state_dict_from_url(url, map_location=None):
    """Try to load the checkopint from the given url.

    """
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=map_location
        )
        return state_dict
    except Exception:
        print('Not able to load state dict from %s' % (url))
        print('Retrying with http:// prefix')
    try:
        url = url.replace('https', 'http')
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=map_location
        )
        return state_dict
    except Exception:
        print('Not able to load state dict from %s' % (url))

    # in this case downloading the pre-trained model was not possible
    # notify the user and return
    return {'state_dict': None}


def _maybe_expand_batchnorm_weights(model_dict, state_dict, num_splits):
    """Expands the weights of the BatchNorm2d to the size of SplitBatchNorm.

    """
    running_mean = 'running_mean'
    running_var = 'running_var'

    for key, item in model_dict.items():
        # not batchnorm -> continue
        if not running_mean in key and not running_var in key:
            continue

        state = state_dict.get(key, None)
        # not in dict -> continue
        if state is None:
            continue
        # same shape -> continue
        if item.shape == state.shape:
            continue

        # found running mean or running var with different shapes
        state_dict[key] = state.repeat(num_splits)

    return state_dict


def _filter_state_dict(state_dict, remove_model_prefix_offset: int = 1):
    """Makes the state_dict compatible with the model.
    
    Prevents unexpected key error when loading PyTorch-Lightning checkpoints.
    Allows backwards compatability to checkpoints before v1.0.6.

    """

    prev_backbone = 'features'
    curr_backbone = 'backbone'

    new_state_dict = {}
    for key, item in state_dict.items():
        # remove the "model." prefix from the state dict key
        key_parts = key.split('.')[remove_model_prefix_offset:]
        # with v1.0.6 the backbone of the models will be renamed from
        # "features" to "backbone", ensure compatability with old ckpts
        key_parts = \
            [k if k != prev_backbone else curr_backbone for k in key_parts]

        new_key = '.'.join(key_parts)
        new_state_dict[new_key] = item

    return new_state_dict


def load_from_state_dict(model,
                         state_dict,
                         strict: bool = True,
                         apply_filter: bool = True,
                         num_splits: int = 0):
    """Loads the model weights from the state dictionary.

    """
    # step 1: filter state dict
    if apply_filter:
        state_dict = _filter_state_dict(state_dict)

    # step 2: expand batchnorm weights
    state_dict = \
        _maybe_expand_batchnorm_weights(model.state_dict(), state_dict, num_splits)

    # step 3: load from checkpoint
    model.load_state_dict(state_dict, strict=strict)
