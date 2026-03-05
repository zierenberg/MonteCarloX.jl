# Weights

Weights determine relative probabilities in equilibrium workflows.
Algorithms only need **differences in log weight**, so the representation can stay flexible.

## 1) Canonical ensemble: `BoltzmannLogWeight`

For energy `E`, the log weight is `-βE`.

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
alg = Metropolis(rng; β=0.5)  # internally uses BoltzmannLogWeight(0.5)
```

Use this for standard fixed-temperature sampling.

## 2) Tabulated weights: `BinnedLogWeight`

Use when the weight is not known analytically, or when adapting it online (multicanonical / Wang-Landau).

```julia
using MonteCarloX

lw = BinnedLogWeight(-20:2:20, 0.0)
lw[0] = 1.5
value = lw(0)

lw_zero = zero(lw)
```

`BinnedLogWeight` supports discrete and continuous bin definitions (including multidimensional bin tuples).

## How to choose

- Known canonical target: `BoltzmannLogWeight`
- Exploratory generalized-ensemble run: `BinnedLogWeight` + `Multicanonical` or `WangLandau`

## API reference

```@docs
BoltzmannLogWeight
BinnedLogWeight
Base.zero(::BinnedLogWeight)
```
