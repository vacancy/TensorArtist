# TensorArtist

## How-to-use
1. Install tensorflow, version v1.0 is required.
2. Set some environs:

```
export PATH=PATH_TO_TENSOR_ARTIST/bin:$PATH
export TART_DIR_DATA=PATH_TO_DATA_DIR
export TART_DIR_DUMP=PATH_TO_DUMP_DIR
```

3. Goto examples/xxx and run: `tart-train desc_xxx.py -d cpu`

## Design Philosophy
1. All experiments description contained in a single file.
2. All parameters stored in global storage, called "core.Environ"
3. All training/testing process based on a container, called "nn.Env"

A good example is better than a thousand-line documentation, feel free to check examples/ to see how it works.


## Thanks:
Contributors:
+ Honghua Dong @dhh1995

And other open-source framework authors, including:
+ Tensorpack(https://github.com/ppwwyyxx/tensorpack) by Yuxin Wu @ppwwyyxx

