

## The idea

Which **experimental configuration** <img src="https://render.githubusercontent.com/render/math?math=\psi"> yields the largest expected gain in information?
The utility <img src="https://render.githubusercontent.com/render/math?math=U(\psi)"> can be expressed as the exected reduction in entropy between the prior and the posterior, i.e.,

<img src="https://render.githubusercontent.com/render/math?math=U(\psi) = \mathbb{E}\left[ \mathbb{H}\left[p(\vartheta)\right] - \mathbb{H}\left[p(\vartheta\vert x,\psi)\right] \right] \propto \mathbb{E}_{p(\vartheta,x\vert\psi)}\left[\log\frac{p(\vartheta\vert x,\psi)}{p(\vartheta)}\right]">.

We seek to obtain the experimental configuration which maximizes the utility: <img src="https://render.githubusercontent.com/render/math?math=\psi^* = \text{argmax}_\psi U(\psi)">

**Problem**: for every evaluation of the utility, the simulator needs to be called because the
expectation depends on  <img src="https://render.githubusercontent.com/render/math?math=p(\vartheta,x\vert\psi)">. Slow and cumbersome!
