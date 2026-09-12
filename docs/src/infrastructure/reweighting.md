# Reweighting

Reweighting reuses **one** set of samples to estimate observables in **many** ensembles.
If we drew ``x_i`` from a source distribution ``q`` but want expectations under a target
``p``, we do not resample — we attach a weight ``w_i = p(x_i)/q(x_i)`` to each draw. This
is the same importance-sampling identity behind [Basic Sampling](../algorithms/basic_sampling.md), used
here deliberately to move *between* ensembles: canonical simulations reweighted to nearby
temperatures (Ferrenberg–Swendsen), or a flat [multicanonical](../algorithms/multicanonical.md) run
reweighted to any ``\beta``.

MonteCarloX forms the weights in log space and returns an [`ImportanceWeights`](@ref)
object; `StatsBase.weights(iw)` exposes them as StatsBase weights for weighted statistics,
[`ess`](@ref) reports how many effective independent samples remain, and
[`log_normalization`](@ref) estimates ``\log(Z_p/Z_q)`` — the free-energy difference
between the two ensembles.

## One sample set, a family of ensembles

Draw ``x`` once from ``q = \mathcal N(0,1)`` and reweight to the exponentially *tilted*
targets ``p_a(x) \propto q(x)\,e^{a x}`` — which are just ``\mathcal N(a,1)``. A single
sample set yields ``\langle x\rangle_{p_a}`` and the free energy for every ``a``, with
exact checks ``\langle x\rangle = a`` and ``\log(Z_{p_a}/Z_q) = a^2/2``.

```@example rw
using MonteCarloX, Distributions, StatsBase, Random
rng = Xoshiro(1)
q   = Normal(0, 1)
xs  = rand(rng, q, 200_000)

tilt(a) = x -> logpdf(q, x) + a * x        # target ∝ q(x)·e^{ax}  =  Normal(a, 1)

map([0.0, 1.0, 2.0, 3.0]) do a
    iw = reweight(xs, q, tilt(a))          # source = q, target = tilted density
    (; a,
       mean_x = round(mean(xs, weights(iw)); digits = 3),   # exact: a
       logZ   = round(log_normalization(iw); digits = 3),   # exact: a²/2
       ess    = round(Int, ess(iw)))
end
```

The estimates track the exact values, but the effective sample size collapses as ``a``
grows: the tilted target lives in the tail of ``q``, where few draws land. That collapse
is the signal to *generate* samples where the target has weight instead of reweighting
from afar — the job of importance-sampled [Metropolis](../algorithms/metropolis.md) or a flat
[multicanonical](../algorithms/multicanonical.md) run, which a single pass can then reweight to the
whole family at once.

## Multiple histograms and WHAM

`wham(histograms, sources)` is the multiple-histogram extension of reweighting. WHAM
means **weighted histogram analysis method**: each histogram ``H_i(x)`` contains counts
from a source distribution with unnormalized log-weight ``b_i(x)``. The shared density
of states ``g(x)`` and the source normalizations ``Z_i`` are obtained from

``g(x) = \frac{\sum_i H_i(x)}{\sum_i N_i\exp(b_i(x)-\log Z_i)}``,
``Z_i = \sum_x g(x)\exp(b_i(x))``.

MonteCarloX solves these equations self-consistently in log space. The ``Z_i`` values
are not manually matched between histograms; they are nuisance normalizations fitted
by the iteration. The current implementation performs exactly `n_iters` iterations:
there is no automatic convergence test or early stopping. `n_iters=1000` is a default
iteration count, not a sample count; it can be changed by the caller after checking
convergence for the problem at hand. Poor overlap, rather than the iteration count,
is the main reason WHAM estimates become unreliable.

The source need not be Boltzmann. Any ensemble or callable log-weight can define a
biased source, provided all histograms use the same coordinate and bins:

```julia
E, log_g = wham(histograms, [source_1, source_2, source_3])
```

Standard WHAM is a **nonparametric** estimate: it assigns one density value to each
observed bin. It could be extended without imposing a physical parametric form by
regularizing the discrete log-density, for example with penalties on its first or
second differences. Such a model would still learn one value per bin, but would use
smoothness to stabilize sparsely sampled regions.

Another option is a flexible function approximator such as a neural network,
``log g(x; θ)``. This is not automatically a physical parametric model: the network
can act as a high-capacity nonparametric approximation. The same biased likelihood
would fit it to multiple histograms, or eventually to raw observations together with
their source log-weights. That route requires careful choices for positivity,
normalization, regularization, optimization, and uncertainty estimates, and can
extrapolate badly outside the observed overlap.

Both directions belong at the boundary between reweighting and inference. They are
promising extensions, but the current histogram WHAM remains the assumption-light
baseline and should be the reference against which regularized or neural estimates
are checked.

## API reference

```@docs
reweight
ImportanceWeights
ess
log_normalization
wham
```
