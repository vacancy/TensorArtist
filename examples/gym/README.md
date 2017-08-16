# Gym (OpenAI reinforcement learning framework) examples

## Basic A3C on Atari games: `desc_a3c_atari_BreakoutV0.py`

```
tart-train desc_a3c_atari_BreakoutV0.py -d 0 
tart-demo desc_a3c_atari_BreakoutV0.py -d 0 [-w xxx.snapshot.pkl] [-e last]
```

This reproduction of A3C is based on ppwwyyxx's reproduction in his TensorPack framework.
Credit to : https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym


## Basic A3C-Continous on Box2D environment: `desc_a3cc_box2d_LunarLanderContinuousV2.py`

This model does not follows the original settings in DeepMind's paper, which use:
1. LSTM model.
2. Episode-as-a-batch update.
3. Gaussian distribution.

In this model, we included several tricks for the training:
1. Truncated Laplacian distribution for policy.
2. Positive advantage only update.

Details can be found in the code.


## Cross-Entropy Method for OpenAI-Gym

This implementation provides a Tensorflow-based CEM optimizer. For memory saving concearn, the optimizer actually does
all optimization on CPU (e.g. storing all parameters and evolve the population using numpy). The most critical
advantage over a pure numpy-based CEM is that you can now use Tensorflow operators to build your neural network.

Note that all optimization-related stuff is done on CPU. So typically you should just use CPU as device.

```
tart-train desc_cem_classic_CartPoleV0 -d cpu
tart-demo desc_cem_classic_CartPoleV0 -d cpu [-w xxx.snapshot.pkl] [-e last]
```

## Evolution Strategies as black-box optimizer for OpenAI-Gym

Simple serial reproduction of Evolution Strategy: [1703.03864] Evolution Strategies as a Scalable Alternative to
Reinforcement Learning (https://arxiv.org/abs/1703.03864).

Similar to CEM optimizer, the ESOptimizer works on CPU, but you can still use Tensorflow to build your network.

Note that the implementation is *NOT* scalable, it does actually serial data collection (not distributed). You should
only this code for simple tests. For the official distributed implementation: refer to
https://github.com/openai/evolution-strategies-starter

```
tart-train desc_es_classic_CartPoleV0 -d cpu
tart-demo desc_es_classic_CartPoleV0 -d cpu [-w xxx.snapshot.pkl] [-e last]
```

## Trust-Region Policy Optimization and Generalized Advantage Estimation

Parallel (multi-threading only) implementation for TRPO and GAE: 
[1502.05477] Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477); 
[1506.02438] High-Dimensional Continuous Control Using Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438).

```
tart-train desc_trpo_gae_box2d_LunarLanderContinuousV2.py -d cpu
tart-demo desc_trpo_gae_box2d_LunarLanderContinuousV2.py -d cpu [-w xxx.snapshot.pkl] [-e last]
```

Thank you authors of the following repos and blogs:
- https://github.com/kvfrans/parallel-trpo
- https://github.com/steveKapturowski/tensorflow-rl
- http://kvfrans.com/speeding-up-trpo-through-parallelization-and-parameter-adaptation

