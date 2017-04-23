# -*- coding:utf8 -*-

from tartist import image, random
from tartist.app import rl
import numpy as np


def r(m):
    return np.pad(
        image.resize_minmax(m.current_state, 128, interpolation='NEAREST'),
        pad_width=[(10, 10), (10, 10), (0, 0)], mode='constant'
    )


def main():
    m = rl.custom.MazeEnv(enable_noaction=False, visible_size=7)
    
    m.restart()
    demo = [r(m)]
    for i in range(19):
        a = random.choice(4)
        m.action(a)
        demo.append(r(m))
   
    i = image.image_grid(demo, ['5v', '4h'])
    image.imshow('Maze', i) 


if __name__ == '__main__':
    main()

