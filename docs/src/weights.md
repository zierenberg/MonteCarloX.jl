# Weights

In equilibrium Monte Carlo, a log weight defines relative probabilities.
Algorithms consume log-weight differences through acceptance rules.

## Canonical weight

`BoltzmannLogWeight(β)` is the default canonical choice.

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
alg = Metropolis(rng; β=0.5)
```

## Mutable/tabulated weight

For generalized ensembles (multicanonical, Wang–Landau),
`BinnedLogWeight` or `MutableLogWeight` (if available) store histogram-backed tables that can be updated.

```julia
using MonteCarloX
using StatsBase

edges = (collect(-10.0:1.0:10.0),)
h = fit(Histogram, Float64[], edges)
lw = BinnedLogWeight(edges, 0.0)

# or initialize directly from edges with a constant value
lw0 = BinnedLogWeight(-10.0:1.0:10.0, 0.0)

# edges can be any sorted vector/range
edges_vec = collect(-5.0:0.5:5.0)
lw1 = BinnedLogWeight(edges_vec, 0.0)

lw[0.3] = 1.2
w = lw(0.3)
```

## API reference

```@docs
BoltzmannLogWeight
BinnedLogWeight
Base.zero(::BinnedLogWeight)
```
