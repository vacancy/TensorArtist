# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, get_logger
from tartist.core.utils.cli import load_desc, parse_devices
from tartist.nn import Env, train

import argparse
import tensorflow as tf

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='desc', help='The description file module')
parser.add_argument('-w', '--initial-weight', dest='initial_weights_path', default=None,
                    help='The pickle containing initial weights, default value can be set in env')
parser.add_argument('-d', '--dev', dest='devices', default=[], nargs='+',
                    help='The devices trainer will use, default value can be set in env')
parser.add_argument('-r', '--root', dest='root', default=None, help='Data dump root')

parser.add_argument('--continue', dest='continue_flag', default=False, action='store_true',
                    help='Whether to continue, if true, continue from the last epoch')
parser.add_argument('--continue-from', dest='continue_from', default=-1, type=int,
                    help='Continue from the given epoch')
parser.add_argument('--quiet', dest='quiet', default=False, action='store_true', help='Quiet run')
parser.add_argument('--queue', dest='use_queue', default=False, action='store_true', help='Use input queues')
args = parser.parse_args()


def main():
    desc = load_desc(args.desc)
    devices = parse_devices(args.devices)
    assert len(devices) > 0, 'Must provide at least one devices'

    env_cls = getattr(desc, '__trainer_env_cls__', train.TrainerEnv)
    env = env_cls(Env.Phase.TRAIN, devices[0])
    env.flags.update(**get_env('trainer.env_flags', {}))
    if len(devices) > 1:
        env.set_slave_devices(devices[1:])

    with env.as_default(activate_session=False):
        if args.use_queue:
            logger.warn('Using input queue for training is now experimental')
            with env.use_input_queue():
                desc.make_network(env)
        else:
            desc.make_network(env)
        desc.make_optimizer(env)

        # debug outputs
        for k in tf.get_default_graph().get_all_collection_keys():
            s = tf.get_collection(k)
            names = ['Collection ' + k] + sorted(['\t{}'.format(v.name) for v in s])
            logger.info('\n'.join(names))

    nr_iters = get_env('trainer.nr_iters', get_env('trainer.epoch_size', 1) * get_env('trainer.nr_epochs', 0))
    trainer_cls = getattr(desc, '__trainer_cls__', train.SimpleTrainer)
    trainer = trainer_cls(nr_iters, env=env, data_provider=desc.make_dataflow_train, desc=desc)
    trainer.set_epoch_size(get_env('trainer.epoch_size', 1))

    from tartist.plugins.trainer_enhancer import snapshot
    if args.continue_flag:
        assert args.continue_from == -1 and args.initial_weights_path is None
        snapshot.enable_snapshot_loading_after_initialization(trainer, continue_last=True)
    elif args.continue_from is not -1:
        assert args.initial_weights_path is None
        snapshot.enable_snapshot_loading_after_initialization(trainer, continue_from=args.continue_from)
    elif args.initial_weights_path is not None:
        snapshot.enable_weights_loading_after_intialization(trainer, weights_fpath=args.initial_weights_path)

    desc.main_train(trainer)


if __name__ == '__main__':
    main()

