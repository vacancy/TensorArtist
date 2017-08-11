# Gym (OpenAI reinforcement learning framework) examples

## Basic A3C on Atari games: `desc_a3c_atari_BreakoutV0.py`

```
tart-train desc_a3c_atari_BreakoutV0.py -d 0 
tart-demo desc_a3c_atari_BreakoutV0.py -d 0 -w xxx.snapshot.pkl
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

```
tart-train desc_cem_classic_CartPoleV0 -d cpu
tart-demo desc_cem_classic_CartPoleV0 -d cpu [-w xxx.snapshot.pkl] [-e last]
```