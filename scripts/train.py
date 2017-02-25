# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, load_env, get_logger, register_event
from tartist.core.utils.cli import load_desc, parse_devices
from tartist.nn import Env, train
import argparse

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='desc', help='The description file module')
parser.add_argument('-w', '--initial-weight', dest='initial_weights_path', default=None,
                    help='The pickle containing initial weights, default value can be set in env')
parser.add_argument('-d', '--dev', dest='devices', default=[], nargs='*',
                    help='The devices trainer will use, default value can be set in env')
parser.add_argument('-r', '--root', dest='root', default=None, help='Data dump root')

parser.add_argument('--continue', dest='continue_flag', default=False, action='store_true',
                    help='Whether to continue, if true, continue from the last epoch')
parser.add_argument('--continue-from', dest='continue_from', default=0, type=int,
                    help='Continue from the given epoch')
parser.add_argument('--quiet', dest='quiet', default=False, action='store_true', help='Quiet run')
args = parser.parse_args()


def main():
    desc = load_desc(args.desc)
    devices = parse_devices(args.devices)

    env = train.TrainerEnv(Env.Phase.TRAIN, devices[0])
    if len(devices) > 1:
        env.set_slave_devices(devices[1:])

    with env.as_default():
        desc.make_network(env)
        desc.make_optimizer(env)

    # debug outputs
    for k, s in env.network.get_all_collections().items():
        names = [k] + sorted(['\t{}'.format(v.name) for v in s])
        logger.info('\n'.join(names))

    trainer = train.SimpleTrainer(env, data_provider=desc.make_dataflow)
    def print_loss(trainer, inp, out):
        loss = out.get('loss', 'N/A')
        logger.info('iter={}: loss={}'.format(trainer.iter, loss))
    register_event('trainer', 'iter:after', print_loss)
    trainer.train()


if __name__ == '__main__':
    main()

