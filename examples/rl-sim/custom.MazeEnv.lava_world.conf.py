# -*- coding:utf8 -*-

import itertools


def make():
    from tartist.app.rl.simulator import automake
    game = automake('custom.MazeEnv', map_size=15)
    obstacles = itertools.chain(
            [(i, 7) for i in range(15) if i not in (3, 11)], 
            [(7, i) for i in range(15) if i not in (3, 11)]
    )
    game.restart(obstacles=obstacles, start_point=(3, 3), finish_point=(11, 11))

    return game

name = 'custom.MazeEnv'
action_names = ['NOOP', 'UP', 'RIGHT', 'DOWN', 'LEFT']
action_keys = [32, 65362, 65363, 65364, 65361]
quit_action_key = 113

# default min size is a tuple (min_dim, max_dim)
default_winsize = (600, 800)
# ignore no-action (action=0) in logging
mute_noaction = True

