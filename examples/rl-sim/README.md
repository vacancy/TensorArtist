# RL environment simulation:

## Usage:

```
tart-rl-sim [-fps FPS] [-rfps RFPS] [-winsize MINDIM MAXDIM] [-record] cfg
```

You need a configuration file `.conf.py` to run it.

To run some defined games, try:
```
tart-rl-sim custom.MazeEnv.lava_world.conf.py -fps 0
tart-rl-sim gym.Enduro-v0.conf.py -fps 24
```

To run autoconf, try:
```
tart-rl-sim-autoconf custom.MazeEnv
tart-rl-sim-autoconf gym.Enduro-v0
```

+ FPS means the FPS you control it, if it's an interactive game (always wait for user control, try `-fps 0`.
+ RFPS is the render fps. You should make sure that RFPS > FPS
+ If you enable `-record`, a replay file will be put in `replays/{game_name}.{time}.replay.pkl` after you finish your
episode. It is useful for recording human plays. See `tartist.rl.simulator.pack` for packed replay.

## Configuration file

You can simply see how `.conf.py` files are organized. It contains `make` function, and several variables defining
the controller. To make highly-customized file (customize the game settings), you can first generate one with autoconf,
and do customization (basically, rewrite `make` function).

For examples, turn to: `custom.MazeEnv.lava_world.conf.py`
