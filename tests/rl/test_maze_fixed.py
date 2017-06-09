# -*- coding:utf8 -*-

from tartist import image, random
from tartist.app import rl
import numpy as np
import itertools


def r(m):
    return np.pad(
        image.resize_minmax(m.current_state, 128, interpolation='NEAREST'),
        pad_width=[(10, 10), (10, 10), (0, 0)], mode='constant'
    )


def main():
    m = rl.custom.MazeEnv(map_size=15, enable_noaction=True, visible_size=None) 
    obstacles = itertools.chain(
            [(i, 7) for i in range(15) if i not in (3, 11)], 
            [(7, i) for i in range(15) if i not in (3, 11)]
    )
    m.restart(obstacles=obstacles, start_point=(3, 3), finish_point=(11, 11))

    demo = [r(m)]
    for i in range(19):
        a = random.choice(4)
        m.action(a)
        demo.append(r(m))
   
    i = image.image_grid(demo, ['5v', '4h'])
    image.imshow('Maze', i) 


if __name__ == '__main__':
    main()
