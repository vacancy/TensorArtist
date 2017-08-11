# Tensor Artist Examples

For beginners, it's good to check the MNIST example to see how it works. A detailed explaination is also included in
that example.

## A list of examples:

#### Classification models:
+ Basic MNIST example with LeNet: `mnist/desc_mnist.py`
+ Basic MNIST example with batch-normalization: `mnist/desc_mnist_bn.py`
+ Basic CIFAR example with tiny CNN: `cifar/desc_cifar.py`
+ Basic CIFAR example with ResNet: `cifar/desc_cifar_resnet.py`

#### Auto-Encoders:
+ Basic MNIST Auto-Encoder example: `auto-encoder/desc_mnist.py`

#### Generative models:
+ Basic MLP Variational Auto-Encoder (VAE) on MNIST: `generative-model/desc_vae_mnist_mlp_bernoulli_adam.py`
+ Basic MLP Generative Adverserial Netowrk (GAN) on MNIST: `generative-model/desc_gan_mnist_mlp.py`
+ Deep Convolutional Generative Adverserial Netowrk (DCGAN) on MNIST: `generative-model/desc_gan_mnist_cnn.py`
+ Wasserstein GAN (WGAN) on MNIST: `generative-model/desc_wgan_mnist_cnn.py`
+ InfoGAN on MNIST: `generative-model/desc_infogan_mnist_cnn.py`
+ DiscoGAN on edges2shoes: `generative-model/desc_discogan_edges2shoes_cnn.py`
+ Deep Recurrent Attention Write (DRAW) on MNIST: `generative-model/desc_draw_mnist.py`

#### Neural Art algorithms:
+ Neural Style: `neural-art/neural_style.py`
+ Deep Dream: `neural-art/deep_dream.py`

#### Deep Reinforcement learning:
+ Basic A3C on Atari games: `gym/desc_a3c_atari_BreakoutV0.py`
+ Basic A3C on Box2D environment: `gym/desc_a3cc_box2d_LunarLanderContinuousV2.py`
+ Cross-Entropy method as gradient-free optimization for RL environments: `gym/desc_cem_classic_CartPoleV0.py`
+ RL Environment Simulator: `rl-sim/README.md`
