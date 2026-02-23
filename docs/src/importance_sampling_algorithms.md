# Importance Sampling Algorithms

Importance-sampling workflows in MonteCarloX use accept/reject-style updates
to target a specified distribution.
You combine:

- a system that defines state and local updates,
- a log weight (`BoltzmannLogWeight` or tabulated variants),
- an algorithm that accepts/rejects proposals.

## Metropolis-family algorithms

`Metropolis` and `Glauber` share the same log-ratio interface but use
different acceptance rules.

Typical model-side update methods compute a local energy change and call
`accept!` through algorithm helpers.

```julia
using Random
using MonteCarloX
using SpinSystems

rng = MersenneTwister(42)
sys = Ising([16, 16], J=1, periodic=true)
init!(sys, :random, rng=rng)

alg = Metropolis(rng; β=0.4)

for _ in 1:50_000
    spin_flip!(sys, alg)
end

println("acceptance = ", acceptance_rate(alg))
```

## Heat-bath updates

`HeatBath` uses conditional local probabilities instead of accept/reject logic.
This can improve mixing for some model classes.

## Generalized ensembles

- `Multicanonical` targets flat-histogram-like exploration by evolving a tabulated log weight.
- `WangLandau` iteratively updates density-of-states estimates and refinement parameter `f`.

```julia
using Random
using MonteCarloX

rng = MersenneTwister(7)
alg = WangLandau(rng)

# update_weight! and update_f! are called in your simulation loop
```

## API reference

```@docs
AbstractImportanceSampling
AbstractGeneralizedEnsemble
AbstractMetropolis
AbstractHeatBath
Metropolis
Glauber
HeatBath
accept!
acceptance_rate
reset!(alg::AbstractImportanceSampling)
Multicanonical
WangLandau
update_weight!
update_f!
```