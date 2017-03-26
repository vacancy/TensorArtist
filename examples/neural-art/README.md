# Neural Art algorithms:

## Neural Style: `neural_style.py`
A reproduction of [1508.06576] A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576).

Usage:

```
tart neural_style.py [-h] -w WEIGHT_PATH -i IMAGE_PATH -s STYLE_PATH -o
                     OUTPUT_PATH [-d DEVICE] [--iter NR_ITERS]
```

## Deep Dream: `deep_dream.py`
A reproduction of https://github.com/google/deepdream.

Usage:

```
tart deep_dream.py [-h] -w WEIGHT_PATH -i IMAGE_PATH -o OUTPUT_PATH [-e END]
                   [-d DEVICE] [--iter NR_ITERS] [--save-step SAVE_STEP]
```
