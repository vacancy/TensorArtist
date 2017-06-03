# -*- coding:utf8 -*-
# File   : rl-sim.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/22/17
# 
# This file is part of TensorArtist

import argparse
import pickle
import time
import threading
import os.path as osp

from tartist import image
from tartist.app.rl.simulator import Controller, Pack, __quit_action__
from tartist.core import get_logger, io
from tartist.core.utils.imp import load_source


logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('cfg')
parser.add_argument('-fps', '--fps', dest='fps', default=24, type=int, help='Controller fps, 0 if wait-keyboard')
parser.add_argument('-rfps', '--render-fps', dest='rfps', default=100, type=int, help='Render fps')
parser.add_argument('-winsize', '--window-size', dest='winsize', default=None, type=int, nargs=2,
                    help='Window size, of format (min_dim, max_dim)')
parser.add_argument('-record', '--record', dest='record', action='store_true', help='Whether to record the play')
args = parser.parse_args()


def vis_and_action(args, cfg, controller, observation):
    # here, the observation is actually a state
    if hasattr(cfg, 'observe'):
        observation = cfg.observe(observation)
    display = image.resize_minmax(observation, *args.winsize, interpolation='NEAREST')
    controller.update(display)

    if args.fps > 0:
        action = 0
        for i in reversed(range(len(cfg.action_keys))):
            if controller.test_key(cfg.action_keys[i]):
                action = i
                break

        if controller.test_key(cfg.quit_action_key):
            action = __quit_action__

        time.sleep((1.0 / args.fps))
    else:
        action = 0
        key = controller.get_last_key()

        for i in reversed(range(len(cfg.action_keys))):
            if key == cfg.action_keys[i]:
                action = i
                break

        if key == cfg.quit_action_key:
            action = __quit_action__

    return action


def dump_pack(cfg, pack):
    pickleable = pack.make_pickleable()

    name = cfg.name + '-'
    name += time.strftime('%Y%m%d-%H%M%S')
    name += '.replay.pkl'
    io.mkdir('replays')
    with open(osp.join('replays', name), 'wb') as f:
        pickle.dump(pickleable, f, pickle.HIGHEST_PROTOCOL)
    logger.critical('replay written to replays/{}'.format(name))


def main(args, controller):
    cfg = load_source(args.cfg, 'cfg')

    if args.winsize is None:
        args.winsize = cfg.default_winsize

    controller.update_title(cfg.name)
    game = cfg.make()
    is_over = False

    if args.record:
        pack = Pack(cfg)
        pack.reset(game.current_state)
    else:
        pack = None

    while True:
        action = vis_and_action(args, cfg, controller, game.current_state)
        if action == __quit_action__:
            break

        if not is_over:
            reward, is_over = game.action(action)

            if args.record:
                pack.step(action, game.current_state, reward, is_over)

            if not cfg.mute_noaction or args.fps == 0 or action != 0:
                logger.info('Perform action: {}, reward={}.'.format(cfg.action_names[action], reward))

            if is_over:
                logger.critical('Game ended, press q (quit key) to exit.')

    logger.critical('Quit action detected: about to exit.')
    controller.quit()
    if args.record:
        dump_pack(cfg, pack)


if __name__ == '__main__':
    controller = Controller(fps=args.rfps)
    thread = threading.Thread(target=main, args=(args, controller))

    thread.start()
    controller.mainloop()
