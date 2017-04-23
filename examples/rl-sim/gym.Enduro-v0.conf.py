# -*- coding:utf8 -*-


def make():
    from tartist.app.rl.simulator import automake
    game = automake('gym.Enduro-v0')
    # you need to restart your game here
    game.restart()
    return game

name = 'gym.Enduro-v0'
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE']
action_keys = [32, 65362, 65363, 65361, 65364, 121, 117, 105, 111]
quit_action_key = 113

# default min size is a tuple (min_dim, max_dim)
default_winsize = (600, 800)
# ignore no-action (action=0) in logging
mute_noaction = True

