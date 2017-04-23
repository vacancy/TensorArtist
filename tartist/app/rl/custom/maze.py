# -*- coding:utf8 -*-
# File   : maze.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/22/17
#
# This file is part of TensorArtist

from ..base import SimpleRLEnvironBase, DiscreteActionSpace
from .... import random
from ....core.utils.meta import notnone_property
from ....core.utils.shape import get_2dshape
import numpy as np
import collections

__all__ = ['MazeEnv']


class MazeEnv(SimpleRLEnvironBase):
    """
    Create a maze environment.
    :param map_size: A single int or a tuple (h, w), representing the map size.
    :param visible_size: A single int or a tuple (h, w), representing the visible size. The agent will at the center
        of the visible window, and out-of-border part will be colored by obstacle color.
    :param obs_ratio: Obstacle ratio (how many obstacles will be in the map).
    :param random_action_mapping: Whether to enable random action mapping. If true, the result of performing
        every action will be shuffled. If a single bool True is provided, we do random shuffle. Otherwise,
        it should be a list with same length as action space (5 when noaction enabled, 4 otherwise).
    :param enable_noaction: Whether to enable no-action operation.
    :param reward_move: Reward for a valid move.
    :param reward_noaction: Reward for a no-action.
    :param reward_final: Reward when you arrive at the final point.
    :param reward_error: Reward when you perform an invalid move.
    """

    _start_point = None
    _final_point = None
    _shortest_path = None
    _current_point = None
    _canvas = None

    """empty, obstacle, current, final, border"""
    _total_dim = 5
    """opencv format: BGR"""
    _colors = [(255, 255, 255), (0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    _action_delta_valid = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # just default value
    _action_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
    _action_mapping = [0, 1, 2, 3, 4]

    def __init__(self, map_size=14, visible_size=None, obs_ratio=0.3,
                 random_action_mapping=None,
                 enable_noaction=True, reward_move=-1, reward_noaction=0, reward_final=100, reward_error=-2):

        super().__init__()
        self._rng = random.gen_rng()
        self._map_size = get_2dshape(map_size)
        self._visible_size = visible_size
        if self._visible_size is not None:
            self._visible_size = get_2dshape(self._visible_size)

        self._obs_ratio = obs_ratio

        if enable_noaction:
            self._action_space = DiscreteActionSpace(5, action_meanings=['NOOP', 'UP', 'RIGHT', 'DOWN', 'LEFT'])
            self._action_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
            self._action_mapping = [0, 1, 2, 3, 4]
        else:
            self._action_space = DiscreteActionSpace(4, action_meanings=['UP', 'RIGHT', 'DOWN', 'LEFT'])
            self._action_delta = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            self._action_mapping = [0, 1, 2, 3]

        if random_action_mapping is not None:
            if random_action_mapping is True:
                self._rng.shuffle(self._action_mapping)
            else:
                assert len(self._action_mapping) == len(random_action_mapping)
                self._action_mapping = random_action_mapping

        self._enable_noaction = enable_noaction
        self._rewards = (reward_move, reward_noaction, reward_final, reward_error)

    @notnone_property
    def canvas(self):
        """Return the raw canvas (full)"""
        return self._canvas

    @notnone_property
    def start_point(self):
        """Start point (r, c)"""
        return self._start_point

    @notnone_property
    def final_point(self):
        """Finish point (r, c)"""
        return self._final_point

    @notnone_property
    def current_point(self):
        """Current point (r, c)"""
        return self._current_point

    @notnone_property
    def shortest_path(self):
        """One of the shortest paths from start to finish, list of point (r, c)"""
        return self._shortest_path

    @property
    def action_delta(self):
        """Action deltas: the tuple (dy, dx) when you perform action i"""
        return self._action_delta

    @property
    def action_mapping(self):
        """If random action mapping is enabled, return the internal mapping"""
        return self._action_mapping

    @property
    def canvas_size(self):
        """Canvas size"""
        return self.canvas.shape[:2]

    @property
    def map_size(self):
        """Map size"""
        return self._map_size

    @property
    def visible_size(self):
        """Visible size"""
        return self._visible_size

    @property
    def rewards(self):
        """A tuple of 4 value, representing the rewards for each action:
        (Move, Noaction, Arrive final point, Move Err)"""
        return self._rewards

    def _color2label(self, cc):
        for i, c in enumerate(self._colors):
            if np.all(c == cc):
                return i
        raise ValueError()

    def _get_canvas_color(self, yy, xx):
        return self._canvas[yy+1, xx+1]

    def _get_canvas_label(self, yy, xx):
        return self._color2label(self._canvas[yy+1, xx+1])

    def _gen_rpt(self):
        """Generate a random point uniformly"""
        return [self._rng.randint(d) for d in self._map_size]

    def _gen_shortest_path(self, c, start_point, final_point):
        sy, sx = start_point
        fy, fx = final_point

        q = collections.deque()
        v = set()
        d = np.ones(self._map_size, dtype='int32') * 100000
        p = np.zeros(self._map_size + (2, ), dtype='int32')

        q.append((sy, sx))
        v.add((sy, sx))
        d[sy, sx] = 0
        p[sy, sx, :] = -1

        while len(q):
            y, x = q.popleft()
            v.remove((y, x))
            if y == fy and x == fx:
                break
            assert self._get_canvas_label(y, x) < 4

            for dy, dx in self._action_delta_valid:
                yy, xx = y + dy, x + dx
                tt = self._get_canvas_label(yy, xx)
                if tt < 4:
                    dd = 2 if tt == 1 else 1
                    if d[yy, xx] > d[y, x] + dd:
                        d[yy, xx] = d[y, x] + dd
                        p[yy, xx, :] = (y, x)
                        if (yy, xx) not in v:
                            q.append((yy, xx))
                            v.add((yy, xx))

        path = []
        y, x = fy, fx

        while y != -1 and x != -1:
            path.append((y, x))
            y, x = p[y, x]
        return path

    def _fill_canvas(self, c, y, x, v, delta=1):
        y += delta
        x += delta
        c[y, x, :] = self._colors[v]

    def _gen_map(self, obstacles=None, start_point=None, final_point=None):
        canvas = np.empty((self._map_size[0] + 2, self._map_size[1] + 2, 3), dtype='uint8')
        canvas[:, :, :] = self._colors[0]

        # reference
        self._canvas = canvas

        for i in range(self._map_size[0] + 2):
            self._fill_canvas(canvas, i, 0, 4, delta=0)
            self._fill_canvas(canvas, i, self._map_size[1] + 1, 4, delta=0)

        for i in range(self._map_size[1] + 2):
            self._fill_canvas(canvas, 0, i, 4, delta=0)
            self._fill_canvas(canvas, self._map_size[1] + 1, i, 4, delta=0)

        if obstacles is None:
            for i in range(int(self._map_size[0] * self._map_size[1] * self._obs_ratio)):
                self._fill_canvas(canvas, *self._gen_rpt(), v=1)
        else:
            for y, x in obstacles:
                self._fill_canvas(canvas, y, x, v=1)

        self._start_point = start_point or self._gen_rpt()
        while True:
            self._final_point = final_point or self._gen_rpt()
            if self._start_point[0] != self._final_point[0] or self._start_point[1] != self._final_point[1]:
                break

        self._fill_canvas(canvas, *self._start_point, v=2)
        self._fill_canvas(canvas, *self._final_point, v=3)

        path = self._gen_shortest_path(canvas, self._start_point, self._final_point)
        for y, x in path:
            self._fill_canvas(canvas, y, x, v=0)

        self._fill_canvas(canvas, *self._start_point, v=2)
        self._fill_canvas(canvas, *self._final_point, v=3)

        self._shortest_path = path
        self._current_point = self._start_point

    def _refresh_view(self):
        if self._visible_size is None:
            self._set_current_state(self._canvas.copy())
            return

        view = np.empty((self._visible_size[0], self._visible_size[1], 3), dtype='uint8')
        view[:, :, :] = self._colors[1]

        y, x = self._current_point
        ch, cw = self.canvas_size
        vh, vw = self._visible_size

        # visible up, left, down, right
        vu, vl = (vh - 1) // 2, (vw - 1) // 2
        vd, vr = vh - vu, vw - vl
        # visible center y, x
        vcy, vcx = vu, vl

        y0, x0 = max(0, y - vu), max(0, x - vl)
        y1, x1 = min(ch, y + vd), min(cw, x + vr)
        vu, vl = y - y0, x - x0
        vd, vr = y1 - y, x1 - x

        view[vcy-vu:vcy+vd, vcx-vl:vcx+vr, :] = self._canvas[y-vu:y+vd, x-vl:x+vr, :]
        self._set_current_state(view)

    def _get_action_space(self):
        return self._action_space

    def _action(self, action):
        if self._enable_noaction and action == 0:
            return self._rewards[1], False

        dy, dx = self._action_delta[self._action_mapping[action]]
        y, x = self._current_point
        y += dy
        x += dx

        if self._get_canvas_label(y, x) in (1, 4):
            return self._rewards[3], False

        if y == self._final_point[0] and x == self._final_point[1]:
            reward = self._rewards[2]
            is_over = True
        else:
            reward = self._rewards[0]
            is_over = False

        self._fill_canvas(self._canvas, *self._current_point, v=0)
        self._current_point = (y, x)
        self._fill_canvas(self._canvas, *self._current_point, v=2)
        self._refresh_view()

        return reward, is_over

    def restart(self, obstacles=None, start_point=None, final_point=None):
        if start_point is not None and final_point is not None:
            assert start_point[0] != final_point[0] or start_point[1] != final_point[1], 'Invalid start and final point: {} {}'.format(
                    start_point, final_point)
        self._gen_map(obstacles=obstacles, start_point=start_point, final_point=final_point)
        self._refresh_view()

    def _restart(self):
        pass
