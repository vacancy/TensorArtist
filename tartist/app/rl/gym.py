# -*- coding:utf8 -*-
# File   : gym.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/18/17
# 
# This file is part of TensorArtist.

from .base import SimpleRLEnvironBase, ProxyRLEnvironBase
from .base import DiscreteActionSpace, ContinuousActionSpace
from tartist.core import io
from tartist.core import get_logger
from tartist.core.utils.meta import run_once
import threading
import numpy as np
import collections

logger = get_logger(__file__)

try:
    import gym
    import gym.wrappers
except ImportError:
    gym = None

_ENV_LOCK = threading.Lock()


def get_env_lock():
    return _ENV_LOCK

__all__ = [
    'GymRLEnviron',
    'GymAtariRLEnviron', 'GymNintendoWrapper', 'GymMarioRLEnviron',
    'GymHistoryProxyRLEnviron', 'GymPreventStuckProxyRLEnviron'
]


class GymRLEnviron(SimpleRLEnvironBase):
    def __init__(self, name, dump_dir=None, force_dump=False, state_mode='DEFAULT'):
        super().__init__()

        with get_env_lock():
            self._gym = self._make_env(name)

        if dump_dir:
            io.mkdir(dump_dir)
            self._gym = gym.wrappers.Monitor(self._gym, dump_dir, force=force_dump)

        assert state_mode in ('DEFAULT', 'RENDER', 'BOTH')
        self._state_mode = state_mode

    def _make_env(self, name):
        return gym.make(name)

    @property
    def gym(self):
        return self._gym

    def render(self, mode='human', close=False):
        return self._gym.render(mode=mode, close=close)

    def _set_current_state(self, o):
        if self._state_mode == 'DEFAULT':
            pass
        else:
            rendered = self.render('rgb_array')
            if self._state_mode == 'RENDER':
                o = rendered
            else:
                o = (o, rendered)
        super()._set_current_state(o)

    def _get_action_space(self):
        spc = self._gym.action_space

        if isinstance(spc, gym.spaces.discrete.Discrete):
            try:
                action_meanings = self._gym.unwrapped.get_action_meanings()
            except AttributeError:
                if 'Atari' in self._gym.unwrapped.__class__.__name__:
                    from gym.envs.atari.atari_env import ACTION_MEANING
                    action_meanings = [ACTION_MEANING[i] for i in range(spc.n)]
                else:
                    action_meanings = ['unknown{}'.format(i) for i in range(spc.n)]
            return DiscreteActionSpace(spc.n, action_meanings=action_meanings)
        elif isinstance(spc, gym.spaces.box.Box):
            return ContinuousActionSpace(spc.low, spc.high, spc.shape)
        else:
            raise ValueError('Unknown gym space spec: {}.'.format(spc))

    def _action(self, action):
        # hack for continuous control
        if type(action) in (tuple, list):
            action = np.array(action)

        o, r, is_over, info = self._gym.step(action)
        self._set_current_state(o)
        return r, is_over

    def _restart(self):
        o = self._gym.reset()
        self._set_current_state(o)

    def _finish(self):
        self._gym.close()


class GymAtariRLEnviron(GymRLEnviron):
    def __init__(self, name, *args, live_lost_as_eoe=True, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._live_lost_as_eoe = live_lost_as_eoe

    def _action(self, action):
        old_lives = self._gym.unwrapped.ale.lives()
        r, is_over = super()._action(action)
        new_lives = self._gym.unwrapped.ale.lives()
        if self._live_lost_as_eoe and old_lives > new_lives:
            is_over = True
        return r, is_over


GymHistoryProxyRLEnviron_warning = run_once(lambda: logger.warn(
    'GymHistoryProxyRLEnviron ' + GymHistoryProxyRLEnviron.__doc__))
from .utils import HistoryFrameProxyRLEnviron as HistoryFrameProxyRLEnviron_


class GymHistoryProxyRLEnviron(HistoryFrameProxyRLEnviron_):
    """DEPRECATED: (2017-12-23) Use HistoryFrameProxyRLEnviron instead."""

    def __init__(self, *args, **kwargs):
        GymHistoryProxyRLEnviron_warning()
        super().__init__(*args, **kwargs)


class GymPreventStuckProxyRLEnviron(ProxyRLEnvironBase):
    def __init__(self, other, max_repeat, action):
        super().__init__(other)
        self._action_list = collections.deque(maxlen=max_repeat)
        self._insert_action = action

    def _action(self, action):
        self._action_list.append(action)
        if self._action_list.count(self._action_list[0]) == self._action_list.maxlen:
            action = self._insert_action
        r, is_over = self.proxy.action(action)
        if is_over:
            self._action_list.clear()
        return r, is_over

    def _restart(self, *args, **kwargs):
        super()._restart(*args, **kwargs)
        self._action_list.clear()

# https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/wrappers/action_space.py
from .gym_adapter import DiscreteToMultiDiscrete
class GymNintendoWrapper(gym.Wrapper):
    """
        Wrapper to convert MultiDiscrete action space to Discrete

        Only supports one config, which maps to the most logical discrete space possible
    """
    def __init__(self, env):
        super().__init__(env)
        # Nintendo Game Controller
        mapping = {
            0: [0, 0, 0, 0, 0, 0],  # NOOP
            1: [1, 0, 0, 0, 0, 0],  # Up
            2: [0, 0, 1, 0, 0, 0],  # Down
            3: [0, 1, 0, 0, 0, 0],  # Left
            4: [0, 1, 0, 0, 1, 0],  # Left + A
            5: [0, 1, 0, 0, 0, 1],  # Left + B
            6: [0, 1, 0, 0, 1, 1],  # Left + A + B
            7: [0, 0, 0, 1, 0, 0],  # Right
            8: [0, 0, 0, 1, 1, 0],  # Right + A
            9: [0, 0, 0, 1, 0, 1],  # Right + B
            10: [0, 0, 0, 1, 1, 1],  # Right + A + B
            11: [0, 0, 0, 0, 1, 0],  # A
            12: [0, 0, 0, 0, 0, 1],  # B
            13: [0, 0, 0, 0, 1, 1],  # A + B
        }
        self.action_space = DiscreteToMultiDiscrete(self.action_space, mapping)

    def _step(self, action):
        return self.env._step(self.action_space(action))


# Using https://github.com/ppaquette/gym-super-mario/tree/gabegrand
# dhh: use meta-env and change_level to hack restart,
#      old restart migth restore to a non-start intermediate state
class GymMarioRLEnviron(GymRLEnviron):
    def __init__(self, name, dump_dir=None, force_dump=False, state_mode='DEFAULT'):
        super().__init__(name, dump_dir, force_dump, state_mode)

        self._cur_iter = -1

    def _make_env(self, name):
        import ppaquette_gym_super_mario
        from ppaquette_gym_super_mario import wrappers
        name_split = name.split('-')
        if name_split[0] != 'meta':
            prefix, world, level = name_split[:3]
            author, prefix = prefix.split('/')
            suffix = '-'.join(name_split[3:])
            self._env_name = '/'.join([author, '-'.join(['meta', prefix, suffix])])
            self._env_level = (int(world) - 1) * 4 + int(level) - 1
        else:
            self._env_name = name
            self._env_level = None
        env = gym.make(self._env_name)
        # modewrapper = wrappers.SetPlayingMode('algo')
        return GymNintendoWrapper(env)

    def _set_info(self, info):
        self.info = info

    def _action(self, action):
        o, r, is_over, info = self._gym.step(action)
        is_over = info.get('iteration', -1) > self._cur_iter
        self._set_info(info)
        self._set_current_state(o)
        return r, is_over 

    def _restart(self):
        if self._cur_iter < 0:
            self._gym.reset()  # hard mario fceux reset
            if self._env_level is not None:
                self._gym.unwrapped.locked_levels = [False, ] * 32
        else:
            o, _, _, info = self._gym.step(7)  # take one step right
            self._gym.unwrapped.change_level(self._env_level)
        # https://github.com/ppaquette/gym-super-mario/issues/4
        o, _, _, info = self._gym.step(7)  # take one step right
        if info.get('ignore', False):  # assuming this happens only in beginning
            self._cur_iter = -1
            self._gym.close()
            self._restart()
        self._cur_iter = info.get('iteration', -1)
        self._set_info(info)
        self._set_current_state(o)

    def _finish(self):
        pass

    def close(self):
        self._gym.close()
