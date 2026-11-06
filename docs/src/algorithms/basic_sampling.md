# Basic Sampling

In general, Monte Carlo methods are employed to estimate some expectation ``\langle f\rangle_p = \int f(x)\,p(x)\,dx``
given a *target* distribution ``p``.
Often, however, it is not possible to draw samples directly from `p` but require another *source* distribution ``q``. 
How we obtain samples from this source and how we process them roughly categorizes Monte Carlo methods.

| strategy   | source vs. target | what do to with the sample                  |
|:-----------|:------------------|:--------------------------------------------|
| simple     | ``q = p``         | keep with weight ``1``                      |
| importance | ``q \neq p``      | keep with weight ``w = p/q``                |
| rejection  | ``q \neq p``      | keep with prob. ``p/(Mq)``, else reject.    |

For basic sampling techniques, MonteCarloX adds merely a convenience layer for numerically robust reweighting using Distributions.jl for the source (`rand`, `logpdf`) and StatsBase for weighted statistics. 
Understanding how MonteCarloX works for simple sampling still provides a good basis to understand MonteCarloX for (advanced) Markov-Chain Monte Carlo and kinetic Monte Carlo methods.

```@example basic
using MonteCarloX, Distributions, StatsBase, Random
rng = Xoshiro(2026)
nothing # hide
```

## Simple sampling

When the target can be sampled directly, the estimator is just the sample mean (Monte Carlo X is not needed).

```@example basic
p  = Normal(1.0, 0.7)
xs = rand(rng, p, 100_000)
(mean = mean(xs), std = std(xs))
```

## Importance sampling

Usually the interesting targets are easy to *evaluate* but hard to *sample*. 
For example, if our target is ``p(x) \propto e^{-x^4/4}`` then we have no analytical Distribution to sample from. 
Instead, we can draw from a convenient ``q`` and correct each draw by the density ratio. 
With [`reweight`](@ref) we can construct these importance weights in a numerically safe way: It constructs an object that stores one *log* weight per sample, ``g_i = \log p(x_i) - \log q(x_i)`` thereby avoiding overflows.
We can then safely use `StatsBase.weights(iw)` to turn them into `AnalyticWeights`.

```@example basic
q       = Normal(0.0, 1.5)
xs      = rand(rng, q, 100_000)
logp(x) = -x^4 / 4                       # the target, up to its unknown normalization
iw      = reweight(xs, q, logp)          # source = q, target = logp
(mean = mean(xs, weights(iw)), var = var(xs, weights(iw)))
```

MonteCarloX offers additional methods to derive information from these importance weights. 
- [`ess`](@ref) estimates the number of independent target samples.
If `ess(iw) \ll length(xs)`, it warns that ``q`` and ``p`` have insufficient overlap.
- [`log_normalization`](@ref) estimates ``\log(Z_p/Z_q)``. When the source distribution ``q`` is normalized, this reduces to ``\log Z_p`` itself, the quantity importance sampling is often used to estimate, such as Bayesian evidence or statistical-mechanical partition functions.

<br><br>

```@example basic
(ess = round(Int, ess(iw)), logZ = log_normalization(iw))
```

## Rejection sampling

[`RejectionSampling`](@ref), see also the [wikipedia article](https://en.wikipedia.org/wiki/Rejection_sampling), aims to obtain true samples from ``p`` by keeping samples from ``q`` with a proper probability.
Since the probablity needs to be smaller than 1, we cannot work with the simple point-wise ratio between the two distributions. 
Instead, we construct an envelope using a constant `M` so that ``p(x) \le M\,q(x)`` everywhere.
Then we accept a sample ``x\sim q`` with probability ``p(x)/(M q(x))``.
The simplest application of this is to sample the area under a curve `f(x)`.
Importantly, it foreshadows the famous [`MetropolisAlgorithm`](@ref), however, with independent samples.

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

How many samples we accept can be read out with [`acceptance_rate`](@ref), which yields ``Z_p/M`` and is the counterpart to  ``\log Z_p`` from importance weights above.

```@example basic
acceptance_rate(alg)
```

## API reference

```@docs
RejectionSampling
MetropolisAlgorithm
```
