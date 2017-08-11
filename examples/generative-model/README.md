# Generative model examples

+ Basic MLP Variational Auto-Encoder (VAE) on MNIST: `desc_vae_mnist_mlp_bernoulli_adam.py`, a reproduction of [1312.6114] Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114).
+ Basic MLP Generative Adverserial Netowrk (GAN) on MNIST: `desc_gan_mnist_mlp.py`, a reproduction of [1406.2661] Generative Adversarial Networks (https://arxiv.org/abs/1406.2661).
+ Deep Convolutional Generative Adverserial Netowrk (DCGAN) on MNIST: `desc_gan_mnist_cnn.py`, a reproduction of [1511.06434] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434).
+ Wasserstein GAN (WGAN) on MNIST: `desc_wgan_mnist_cnn.py`, a reproduction of [1701.07875] Wasserstein GAN (https://arxiv.org/abs/1701.07875).
+ InfoGAN on MNIST: `desc_infogan_mnist_cnn.py`, a reproduction of [1606.03657] InfoGAN: Interpretable Representation Learning by Information Maximizing\n  Generative Adversarial Nets (https://arxiv.org/abs/1606.03657).
+ DiscoGAN on edges2shoes: `desc_discogan_edges2shoes_cnn.py`, a reproduction of [1703.05192] Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (https://arxiv.org/abs/1703.05192).
+ Deep Recurrent Attention Write (DRAW) on MNIST: `desc_draw_mnist.py`, a reproduction of [1502.04623] DRAW: A Recurrent Neural Network For Image Generation (https://arxiv.org/abs/1502.04623).

## Usage

All these experiments support simply:

```
tart-train desc_xxx.py -d gpu0
tart-demo desc_xxx.py -d gpu0 -w xxx.snapshot.pkl
```

For DiscoGAN, special dataset toolkits (and their usage) available at `TensorArtist/scripts/dataset-tools/pix2pix`.

## Thank

During the reproduction, the author refer to the following repos:

1. https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW
2. https://github.com/ppwwyyxx/tensorpack/tree/master/examples/GAN
3. https://github.com/zxie/vae
4. https://github.com/carpedm20/DCGAN-tensorflow
5. https://github.com/jbornschein/draw
6. https://github.com/carpedm20/DiscoGAN-pytorch
