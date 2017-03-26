# MNIST Example: A tutorial for beginners.

## Basic Usage:

Please make sure that you have set the OS environment variables according to the top-level installation introduction.

```
tart-train desc_mnist.py -d cpu
tart-demo desc_mnist.py -d cpu -w $TART_DUMP/mnist/mnist/snapshots/epoch_<eid>.snapshot.pkl
```

## Some details of the description files and scripts
As state in "design philosophy", all the experiments settings (including but not limited to network structures, 
hyperparameters, data processing strategies...) should be included in a *SINGLE* `desc_*` file.
As we can see in the `desc_mnist.py`, it basically includes three parts, `make_network`, `make_optimizer`, and `__envs__`.

#### Environ (`__envs__`)
The `Environ` is just a single dict. It contains all the parameters/configurations you need to conduct your training process / inference / etc.
For example `envs['dir']['root']` is the root directory for this experiments, and thus all the models, logs of this experiment will be dumped into 
that specific directory.

The reason to introduce a global Environ is that these configs can be accessed anywhere in TensorArtist, simply by using the function `get_env`.
You can easily add new settings to your program, and use it somewhere else. A worth-noting thing is that the function `get_env` provide simplified
access to cascaed dicts, i.e. `get_env('trainer.learning_rate')` is equivalent to `get_env('trainer')['learning_rate']`

#### Network making function (`make_network`)
In the network making function, you are given an `Env` (NOTE!! this object has noting to do with the `Environ` introduced before). An env is similar
to `tf.Graph` is design. If you want to derive more about the Env, feel free to take a look at `tartist/nn/graph/env.py` to find more details.

A worth noting thing in this function is the usage of `DataParallelController`, or abbr. `dpc`. It provides
an automatical way to perform data-parallel computation acrossing different GPUs. You can see the example
for MNIST to see the basic usage. We also include a very detailed explaination on how it works in `tartist/nn/graph/env.py`.

In this function, you can call `O.xxx` to build your computation graph. All O.xxx functions deals with `VarNode` (a wrapper of `tf.Tensor`), 
and you can convert between them using `O.as_varnode` and `O.as_tftensor`. To find a complete list of available ops, see `tartist/nn/opr`.

#### Optimizer making function (`make_optimizer`)
When you want to train a neural network on some dataset, you also need to provide an optimizer. You can see the example to see how to add 
grad processors.

#### Data provider
Another thing not explicitly included in the description file is how to provide the data for training. Actually we included this function from
`data_provider_mnist.py`. You can see it for detail. The high-level idea is that you need to provide an object called `DataFlow` in your function.
It is just a kind of generator from which we can get data.

#### Mainloop (`main_train`)
In this function, typically we first add some plugins to the trainer depending on our need, and then call `trainer.train`.

### Conclusion: a list of top-level functions / variables in a description file
+ `__envs__` used for environ setting
+ `make_network` used for computation graph building
+ `make_optimizer` used for setting optimizer
+ `make_dataflow_train` used for provide data for the trainer
+ `make_dataflow_inference` used for provide data for the inference plugin (e.g. it will be executed after every epoch to test the model on 
the validation set)
+ `make_dataflow_demo` used for provide input data for the demo
+ `main_train` used to start the trainer, and register extensions for the trainer
+ `demo` used for demonstrate (typically show the image) the result
+ `main_demo` if this function is provided, `main_dataflow_demo` and `demo` will be ignored, you can run your own demo.

Finally, please feel free to check `scripts/train.py` and `scripts/demo.py` to see the detail. I promise they are simple and easy-to-read.


