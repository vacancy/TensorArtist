# -*- coding:utf8 -*-
# File   : rl-sim-autoconf.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/22/17
# 
# This file is part of TensorArtist.

import argparse
import threading
import os.path as osp

from tartist import image
from tartist.core import get_logger
from tartist.core.utils.cli import yes_or_no
from tartist.app.rl import DiscreteActionSpace
from tartist.app.rl.simulator import Controller, automake

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('gamename')
parser.add_argument('-o', '--output', dest='output', help='A conf.py file.', default=None)
parser.add_argument('-winsize', '--window-size', dest='winsize', default=(600, 800), type=int, nargs=2)


__automake_format__ = """
def make():
    from tartist.app.rl.simulator import automake
    game = automake('{}')
    # you need to restart your game here
    game.restart()
    return game\n
"""


def main(args, controller):
    if args.output is None:
        args.output = '{}.conf.py'.format(args.gamename)

    if osp.isfile(args.output):
        logger.warn('Output config file already exists: {}.'.format(args.output))
        if not yes_or_no('Do you want to overwrite?', default='no'):
            controller.quit()
            return

    game = automake(args.gamename)
    game.restart()

    action_space = game.action_space
    assert isinstance(action_space, DiscreteActionSpace)

    action_names = action_space.action_meanings
    action_keys = []

    logger.critical('All action names: {}'.format(action_names))
    logger.critical('Start recording...')

    img = image.resize_minmax(game.current_state, *args.winsize, interpolation='NEAREST')
    controller.update_title(args.gamename)
    controller.update(img)

    for i in range(len(action_names)):
        name = action_names[i]
        logger.info('{}-th action, name={}, waiting...'.format(i, name))
        action = controller.get_last_key()
        action_keys.append(action)
        logger.info('{}-th action, name={}, key={}, done.'.format(i, name, repr(action)))
    logger.critical('Recording end.')

    logger.info('Recoding quit action key')
    quit_key = controller.get_last_key()
    logger.info('quit action, key={}, done.'.format(repr(quit_key)))
    controller.quit()

    with open(args.output, 'w') as f:
        f.write('# -*- coding:utf8 -*-\n\n')
        f.write(__automake_format__.format(args.gamename))
        f.write("name = '{}'\n".format(args.gamename))
        f.write("action_names = {}\n".format(repr(action_names)))
        f.write("action_keys = {}\n".format(repr(action_keys)))
        f.write("quit_action_key = {}\n".format(repr(quit_key)))
        f.write("\n")
        f.write("# default min size is a tuple (min_dim, max_dim)\n")
        f.write("default_winsize = (600, 800)\n")
        f.write("# ignore no-action (action=0) in logging\n")
        f.write("mute_noaction = True\n")
        f.write("\n")
    logger.critical('Cfg file written to {}'.format(args.output))


if __name__ == '__main__':
    controller = Controller()
    thread = threading.Thread(target=main, args=(parser.parse_args(), controller))

    thread.start()
    controller.mainloop()
