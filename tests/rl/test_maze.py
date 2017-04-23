# -*- coding:utf8 -*-

from tartist import image, random
from tartist.app import rl

m = rl.custom.MazeEnv(enable_noaction=False, visible_size=7)

m.restart()
gg = [m.current_state]
for i in range(19):
    a = random.choice(4)
    m.action(a)
    gg.append(m.current_state)

i = image.image_grid(gg, ['10h'])
image.imshow('gg', i)

