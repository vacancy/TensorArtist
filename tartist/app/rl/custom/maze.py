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
import itertools

__all__ = ['MazeEnv', 'CustomLavaWorldEnv']


class MazeEnv(SimpleRLEnvironBase):
    """
    Create a maze environment.

    :param map_size: A single int or a tuple (h, w), representing the map size.
    :param visible_size: A single int or a tuple (h, w), representing the visible size. The agent will at the center
        of the visible window, and out-of-border part will be colored by obstacle color.
    :param obs_ratio: Obstacle ratio (how many obstacles will be in the map).
    :param enable_path_checking: Enable path computation in map construction. Turn it down only when you are sure about 
        valid maze.
    :param random_action_mapping: Whether to enable random action mapping. If true, the result of performing
        every action will be shuffled. _checkingIf a single bool True is provided, we do random shuffle. Otherwise,
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
    _distance_mat = None
    _distance_prev = None
    _quick_distance_mat = None
    _quick_distance_prev = None
    _current_point = None

    _canvas = None
    _origin_canvas = None

    """empty, obstacle, current, final, border"""
    _total_dim = 5
    """opencv format: BGR"""
    _colors = [(255, 255, 255), (0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    _action_delta_valid = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # just default value
    _action_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
    _action_mapping = [0, 1, 2, 3, 4]

    def __init__(self, map_size=14, visible_size=None, obs_ratio=0.3, enable_path_checking=True,
                 random_action_mapping=None, 
                 enable_noaction=False, reward_move=-1, reward_noaction=0, reward_final=10, reward_error=-2):

        super().__init__()
        self._rng = random.gen_rng()
        self._map_size = get_2dshape(map_size)
        self._visible_size = visible_size
        self._enable_path_checking = enable_path_checking
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
    def origin_canvas(self):
        """Return the original canvas (at time 0, full)"""
        return self._origin_canvas

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

    @notnone_property
    def quick_distance_mat(self):
        """Distance matrix: this is done during the first run of SPFA, so if you ensure
        that all valid points are in the same connected component, you can use it"""
        return self._quick_distance_mat

    @notnone_property
    def quick_distance_prev(self):
        """Distance-prev matrix: see also `quick_distance_mat`"""
        return self._quick_distance_mat

    @property
    def distance_mat(self):
        """Distance matrix"""
        self._gen_distance_info()
        return self._distance_mat

    @property
    def distance_prev(self):
        """Distance-prev matrix"""
        self._gen_distance_info()
        return self._distance_prev

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

    def _fill_canvas(self, c, y, x, v, delta=1):
        c[y + delta, x + delta, :] = self._colors[v]

    def _gen_shortest_path(self, c, start_point, final_point):
        sy, sx = start_point
        fy, fx = final_point

        obs_dis = self.canvas_size[0] * self.canvas_size[1]
        q = collections.deque()
        v = set()
        d = np.ones(self._map_size, dtype='int32') * obs_dis * 2
        p = np.zeros(self._map_size + (2, ), dtype='int32')

        q.append((sy, sx))
        v.add((sy, sx))
        d[sy, sx] = 0
        p[sy, sx, :] = -1

        while len(q):
            y, x = q.popleft()
            v.remove((y, x))
            assert self._get_canvas_label(y, x) < 4

            for dy, dx in self._action_delta_valid:
                yy, xx = y + dy, x + dx
                tt = self._get_canvas_label(yy, xx)
                if tt < 4:
                    dd = obs_dis if tt == 1 else 1
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
        return path, d, p

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
        
        if self._enable_path_checking:
            path, d, p = self._gen_shortest_path(canvas, self._start_point, self._final_point)
            for y, x in path:
                self._fill_canvas(canvas, y, x, v=0)

            self._fill_canvas(canvas, *self._start_point, v=2)
            self._fill_canvas(canvas, *self._final_point, v=3)

            self._shortest_path = path
            self._quick_distance_mat = d
            self._quick_distance_prev = p

        self._current_point = self._start_point

        self._origin_canvas = canvas.copy()

    def _gen_distance_info(self):
        path, d, p = self._gen_shortest_path(self.origin_canvas, self._start_point, self._final_point)
        self._distance_mat = d
        self._distance_prev = p

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


class CustomLavaWorldEnv(MazeEnv):
    """A maze similar to Lava World in OpenAI Gym"""

    _empty_canvas = None

    def __init__(self, map_size=15, mode=None, **kwargs):
        kwargs.setdefault('enable_path_checking', False)
        super().__init__(map_size, **kwargs)

        mode = mode or 'ALL'
        assert mode in ('ALL', 'TRAIN', 'VAL')
        h, w = get_2dshape(map_size)

        assert h % 4 == 3 and w % 4 == 3

        self._lv_obstacles = list(itertools.chain(
            [(i, (w-1) // 2) for i in range(h) if i not in ((h-3) // 4, (h-1)//2 + (h+1)//4)],
            [((h-1) // 2, i) for i in range(w) if i not in ((w-3) // 4, (w-1)//2 + (w+1)//4)]
        ))
        self._lv_starts = [(i,j) for i in range(h) for j in range(w) if (i,j) not in self._lv_obstacles]
        if mode == 'ALL':
            self._lv_finals = self._lv_starts.copy()
        elif mode == 'TRAIN':
            self._lv_finals = [(i,j) for i in range(h) for j in range(w)
                    if (i < h // 2 or j < w // 2) and (i,j) not in self._lv_obstacles]
        elif mode == 'VAL':
            self._lv_finals = [(i,j) for i in range(h) for j in range(w)
                    if not (i < h // 2 or j < w // 2) and (i,j) not in self._lv_obstacles]

    @property
    def lv_obstacles(self):
        return self._lv_obstacles

    @property
    def lv_starts(self):
        return self._lv_starts

    @property
    def lv_finals(self):
        return self._lv_finals

    def restart(self, start_point=None, final_point=None):
        if start_point is None:
            i = random.choice(len(self.lv_starts))
            start_point = self.lv_starts[i]
        start_point = tuple(start_point)

        if final_point is None:
            while True:
                j = random.choice(len(self.lv_finals))
                final_point = self.lv_finals[j]
                if start_point != final_point:
                    break
        final_point = tuple(final_point)

        assert start_point != final_point, 'Invalid start and final point: {} {}'.format(
                start_point, final_point)
        
        if self._empty_canvas is None:
            super().restart(self.lv_obstacles, start_point, final_point)
            self._empty_canvas = self._canvas.copy()
            self._fill_canvas(self._empty_canvas, *self._start_point, v=0)
            self._fill_canvas(self._empty_canvas, *self._final_point, v=0)
        else:
            # do partial reload
            self._start_point = start_point
            self._final_point = final_point
            self._current_point = start_point
            self._canvas = self._empty_canvas.copy()
            self._fill_canvas(self._canvas, *self._start_point, v=2)
            self._fill_canvas(self._canvas, *self._final_point, v=3)
            self._origin_canvas = self._canvas.copy()
            self._refresh_view()

