[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/amortized-experimental-design/HEAD)

## The idea

Which **experimental configuration** <img src="https://render.githubusercontent.com/render/math?math=\psi"> yields the largest expected gain in information?
The utility <img src="https://render.githubusercontent.com/render/math?math=U(\psi)"> can be expressed as the exected reduction in entropy between the prior and the posterior, i.e.,

<img src="https://render.githubusercontent.com/render/math?math=U(\psi) = \mathbb{E}\left[ \mathbb{H}\left[p(\vartheta)\right] - \mathbb{H}\left[p(\vartheta\vert x,\psi)\right] \right] \propto \mathbb{E}_{p(\vartheta,x\vert\psi)}\left[\log\frac{p(\vartheta\vert x,\psi)}{p(\vartheta)}\right]">.

This quantity is especially challenging because the posterior is intractable in most practical applications. We can however, draw samples from the likelihood model (our simulator).
We seek to obtain the experimental configuration which maximizes the utility: <img src="https://render.githubusercontent.com/render/math?math=\psi^* = \text{argmax}_\psi U(\psi)">

**Problem**: for every evaluation of the utility, the simulator needs to be called because the
expectation depends on  <img src="https://render.githubusercontent.com/render/math?math=p(\vartheta,x\vert\psi)">. Slow and cumbersome!

**Proposal:** Reweigh the samples from the joint <img src="https://render.githubusercontent.com/render/math?math=p(\vartheta,x,\psi)"> to approximate <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{p(\vartheta,x\vert\psi)}\left[\log\frac{p(\vartheta\vert x,\psi)}{p(\vartheta)}\right]"> with several ratio estimators that can be trained on samples from the joint alone!
In doing so, we can estimate the EIG for every experimental configuration by reusing presimulated samples for specific experimental configurations. In addition, this specification allows us to maximize the expected information gain through gradient ascent by simply backpropagating through the neural networks. For more details check the notebook!

## License

See LICENSE file.
