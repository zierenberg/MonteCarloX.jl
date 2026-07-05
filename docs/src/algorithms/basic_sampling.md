# Basic Sampling

Monte Carlo estimation asks for an expectation ``\langle f\rangle_p = \int f(x)\,p(x)\,dx``
under a *target* distribution ``p``. What differs between methods is how we obtain
samples and what we do with each one, given a *source* distribution ``q`` that we can
actually draw from.

| strategy   | source vs. target | what we do with a draw                      |
|:-----------|:------------------|:--------------------------------------------|
| simple     | ``q = p``         | keep it, weight ``1``                       |
| importance | ``q \neq p``      | keep it, weight ``w = p/q``                 |
| rejection  | ``q \neq p``      | keep it with prob. ``p/(Mq)``, else discard |

MonteCarloX leans on Distributions.jl for the source (`rand`, `logpdf`) and on StatsBase
for weighted statistics; its own contribution is the reweighting bridge — forming the
weights safely in log space and reporting how far they can be trusted.

```@example basic
using MonteCarloX, Distributions, StatsBase, Random
rng = Xoshiro(2026)
nothing # hide
```

## Simple sampling

When the target can be sampled directly, the estimator is just the sample mean.

```@example basic
p  = Normal(1.0, 0.7)
xs = rand(rng, p, 100_000)
(mean = mean(xs), std = std(xs))
```

## Importance sampling

Usually the interesting targets are easy to *evaluate* but hard to *sample*. Take the
quartic density ``p(x) \propto e^{-x^4/4}``: there is no `rand` for it. We draw from a
convenient source ``q`` and correct each draw by the density ratio. [`reweight`](@ref)
holds one *log* weight per sample, ``g_i = \log p(x_i) - \log q(x_i)``, in log space so
nothing overflows; `StatsBase.weights(iw)` turns them into `AnalyticWeights`.

```@example basic
q       = Normal(0.0, 1.5)
xs      = rand(rng, q, 100_000)
logp(x) = -x^4 / 4                       # the target, up to its unknown normalization
iw      = reweight(xs, q, logp)          # source = q, target = logp
(mean = mean(xs, weights(iw)), var = var(xs, weights(iw)))
```

Two by-products come for free. The [`ess`](@ref) reports how many independent target
draws the weighted set is worth — far below `length(xs)` warns that ``q`` and ``p``
overlap too poorly. And [`log_normalization`](@ref) estimates ``\log(Z_p/Z_q))`` — with a
normalized Gaussian ``q`` that is ``\log Z_p`` itself, the quantity importance sampling
exists to reach (Bayesian evidence, partition functions).

```@example basic
(ess = round(Int, ess(iw)), n = length(xs), logZ = log_normalization(iw))
```

## Rejection sampling

Reweighting reuses every draw; rejection sampling instead returns *exact* samples from
``p`` by throwing some away. We trap the target under a scaled proposal, ``p(x) \le M\,q(x)``
everywhere, and accept a draw ``x\sim q`` with probability ``p(x)/(M q(x))``.
[`RejectionSampling`](@ref) is the independent-sample analogue of Metropolis: we own the
proposal loop and call `accept!` on each candidate, with the envelope ``g = M q``.

```@example basic
logM = maximum(logp(x) - logpdf(q, x) for x in range(-6, 6; length = 20_001)) + 1e-6
alg  = RejectionSampling(rng, x -> logpdf(q, x) + logM, logp)   # envelope g = M·q

samples = Float64[]
while length(samples) < 50_000
    x = rand(rng, q)
    accept!(alg, x) && push!(samples, x)
end
(mean = mean(samples), std = std(samples))
```

The accepted fraction is itself meaningful — [`acceptance_rate`](@ref) estimates
``Z_p/M``, rejection sampling's counterpart to the ``\log Z_p`` the importance weights gave.

```@example basic
acceptance_rate(alg)
```

## API reference

```@docs
RejectionSampling
```
