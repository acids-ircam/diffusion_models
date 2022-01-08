# Denoising diffusion probabilistic models

These tutorials explores the new class of generative models based on _diffusion probabilistic models_ [ [ 1 ] ](#ref1). This class of models is inspired by considerations from thermodynamics [ [ 2 ] ](#ref2), but also bears strong ressemblence to _denoising score matching_ [ [ 3 ] ](#ref3), _Langevin dynamics_ and _autoregressive decoding_. We will also discuss the more recent development of _denoising diffusion implicit models_ [ [ 4 ] ](#ref4), which bypass the need for a Markov chain to accelerate the sampling. Stemming from this work, we will also discuss the _wavegrad_ model [ [ 5 ] ](#ref5), which is based on the same core principles but applies this class of models for audio data. 

In order to fully understand the inner workings of diffusion model, we will review all of the correlated topics through tutorial notebooks. These notebooks are available in `Pytorch` or in `JAX` (in the [`jax_tutorials/`](https://github.com/acids-ircam/diffusion_models/tree/main/jax_tutorials) folder), thanks to the great contribution of [Cristian Garcia](https://github.com/cgarciae).

We split the explanation between four detailed notebooks.
1. Score matching and Langevin dynamics.
2. Diffusion probabilistic models and denoising
3. Applications to waveforms with WaveGrad
4. Implicit models to accelerate inference

<a id="ref1"/>

[1] [Ho, J., Jain, A., & Abbeel, P. (2020). _Denoising diffusion probabilistic models_. arXiv preprint arXiv:2006.11239.](https://arxiv.org/pdf/2006.11239)

<a id="ref2"/>

[2] [Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. arXiv preprint arXiv:1503.03585.](https://arxiv.org/pdf/1503.03585)

<a id="ref3"/>

[3] [Vincent, P. (2011). A connection between score matching and denoising autoencoders. Neural computation, 23(7), 1661-1674.](http://www-labs.iro.umontreal.ca/~vincentp/Publications/smdae_techreport_1358.pdf)

<a id="ref4"/>

[4] [Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. arXiv preprint arXiv:2010.02502.](https://arxiv.org/pdf/2010.02502.pdf)

<a id="ref5"/>

[5] [Chen, N., Zhang, Y., Zen, H., Weiss, R. J., Norouzi, M., & Chan, W. (2020). _WaveGrad: Estimating gradients for waveform generation_. arXiv preprint arXiv:2009.00713.](https://arxiv.org/pdf/2009.00713)
