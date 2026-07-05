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

## API reference

```@docs
reweight
ImportanceWeights
ess
log_normalization
```
