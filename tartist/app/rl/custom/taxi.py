# -*- coding:utf8 -*-
# File   : taxi.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/4/17
#
# This file is part of TensorArtist

from ..base import SimpleRLEnvironBase
from .maze import MazeEnv, CustomLavaWorldEnv

__all__ = ['CustomTaxiEnv', 'CustomLavaWorldTaxiEnv']


class CustomTaxiEnv(SimpleRLEnvironBase):
    _start_point = None
    _final_point1 = None
    _final_point2 = None

    def __init__(self, maze_env=None, *args, **kwargs):
        super().__init__()
        if maze_env is None:
            maze_env = MazeEnv(*args, **kwargs)
        assert isinstance(maze_env, MazeEnv)
        self._maze_env = maze_env
        self._phase = 0

    @property
    def maze_env(self):
        return self._maze_env

    @property
    def phase(self):
        return self._phase

    def _get_action_space(self):
        return self._maze_env.action_space

    def restart(self, start_point=None, final_point1=None, final_point2=None):
        self._start_point = start_point
        self._final_point1 = final_point1
        self._final_point2 = final_point2
        self._maze_env.restart(start_point=start_point, final_point=final_point1)
        self._phase = 1
        self._set_current_state(self._maze_env.current_state)

        super().restart()

    def _restart(self):
        pass

    def _action(self, action):
        # Sanity phase check
        assert self._phase in (1, 2), 'Invalid phase'

        reward, is_over = self._maze_env.action(action)
        if is_over:
            if self._phase == 1:
                self._phase = 2
                self._enter_phase2()
                is_over = False

        self._set_current_state(self._maze_env.current_state)
        return reward, is_over

    def _enter_phase2(self):
        self._maze_env.restart(
            obstacles=self._maze_env.obstacles,
            start_point=self._maze_env.current_point,
            final_point=self._final_point2
        )

    def _finish(self):
        self._maze_env.finish()


class CustomLavaWorldTaxiEnv(CustomTaxiEnv):
    def __init__(self, maze_env=None, *args, **kwargs):
        if maze_env is None:
            maze_env = CustomLavaWorldEnv(*args, **kwargs)
        assert isinstance(maze_env, CustomLavaWorldEnv)
        super().__init__(maze_env)

    def _enter_phase2(self):
        self._maze_env.restart(start_point=self._maze_env.current_point, final_point=self._final_point2)

    def _finish(self):
        super()._finish()

        if self._phase == 2 and self._maze_env._current_point == self._maze_env._final_point:
            self.append_stat('success', 1)
        else:
            self.append_stat('success', 0)

