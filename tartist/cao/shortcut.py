# -*- coding:utf8 -*-
# File   : shortcut.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/28/17
# 
# This file is part of TensorArtist.


from .. import image
from .. import random
from ..core import io, get_logger
from ..core import get_env, set_env
from ..core import register_event, trigger_event
from ..core.utils.cli import load_desc, parse_devices, parse_args
from ..app import gan, rl
from ..data import flow, rflow
from ..nn import opr as O
from ..nn.opr import callback_injector as cbi
from ..nn.graph import Env, Network, DataParallelController, get_default_net, get_default_env
from ..nn.graph import Function, VarNode, OprNode, as_varnode, as_tftensor
from ..plugins.trainer_enhancer.snapshot import load_weights_file, load_snapshot_file

import numpy as np
import tensorflow as tf

logger = get_logger()

__all__ = [
    'image', 'random', 'io', 'logger', 'get_logger',
    'get_env', 'set_env',
    'register_event', 'trigger_event',
    'gan', 'rl', 'flow', 'rflow',
    'load_desc', 'parse_devices', 'parse_args', 'load_weights_file', 'load_snapshot_file',
    'O', 'cbi', 'Env', 'Network', 'DataParallelController', 'get_default_env', 'get_default_net',
    'Function', 'VarNode', 'OprNode', 'as_tftensor', 'as_varnode',
    'np', 'tf'
]
