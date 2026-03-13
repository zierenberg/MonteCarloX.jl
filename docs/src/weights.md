# Weights

Weights determine relative probabilities in equilibrium workflows.
Algorithms only need **differences in log weight**, so the representation can stay flexible.

The same callable concept also covers Bayesian targets:

- Bayesian inference: `logweight(theta) = logposterior(theta) = loglikelihood(theta) + logprior(theta)`
- Statistical mechanics: `logweight(x) = -beta * E(x)`

In the sampler API this callable is stored as the algorithm `ensemble`, with
`logweight(alg)` as an equivalent accessor.

For canonical targets,
\[
\pi(x)=\frac{e^{-\beta E(x)}}{Z(\beta)},
\]
so
\[
\log \pi(x) = -\beta E(x) - \log Z(\beta),
\]
and local acceptance decisions depend on
\[
\Delta \log \pi = -\beta\,\Delta E.
\]

## 1) Canonical ensemble: `BoltzmannEnsemble`

For energy `E`, the log weight is `-βE`.

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
alg = Metropolis(rng; β=0.5)  # internally uses BoltzmannEnsemble(0.5)
```

Use this for standard fixed-temperature sampling.

## 2) Tabulated weights: `BinnedObject`

Use when the weight is not known analytically, or when adapting it online (multicanonical / Wang-Landau).

```julia
using MonteCarloX

lw = BinnedObject(-20:2:20, 0.0)
lw[0] = 1.5
value = lw(0)

lw_zero = zero(lw)
```

`BinnedObject` supports discrete and continuous bin definitions (including multidimensional bin tuples).

## How to choose

- Known canonical target: `BoltzmannEnsemble`
- Exploratory generalized-ensemble run: `BinnedObject` + `Multicanonical` or `WangLandau`

## API reference

```@docs
BoltzmannEnsemble
FunctionEnsemble
BinnedObject
get_centers(bo::BinnedObject, dim::Int=1)
Base.values(bo::BinnedObject)
Base.zero(lw::BinnedObject)
```
