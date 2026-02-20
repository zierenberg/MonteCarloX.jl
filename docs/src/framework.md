# Framework: how to think in MonteCarloX

MonteCarloX is designed around **composable parts** instead of one monolithic simulator.
A simulation is built by combining:

1. **System** (`AbstractSystem`): what state you evolve
2. **Weight / rates** (`AbstractLogWeight` or rates): what distribution/dynamics define probability
3. **Algorithm** (`AbstractAlgorithm`): how proposals/events are generated
4. **Update** (`AbstractUpdate` + model-side methods): how accepted events modify state
5. **Measurement** (`Measurement`, `Measurements`): what observables are recorded

This separation is intentional:

- You can keep the **same algorithm** and swap in a different system.
- You can keep the **same system** and change the ensemble via a different log weight.
- You can reuse a **measurement setup** across multiple algorithms.

## Two sampling paradigms

MonteCarloX supports two complementary paradigms:

### Importance sampling

```text
System + LogWeight + Proposal/Accept + Apply + Measure
```

Primary goal: sample from a target stationary distribution.

### Continuous-time sampling

```text
State + Event rates + Sampler + Measure + Apply
```

Primary goal: evolve stochastic dynamics in physical/simulation time.

In continuous-time workflows, the sampler provides `(dt, event)` and advances
time, then measurements are taken at the new time before applying the event
update.

## Minimal importance-sampling workflow

```julia
using Random
using MonteCarloX
using SpinSystems

rng = MersenneTwister(42)
sys = Ising([16, 16], J=1, periodic=true)
init!(sys, :random, rng=rng)

alg = Metropolis(rng; β=0.4)

measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)

for step in 1:100_000
    spin_flip!(sys, alg)
    measure!(measurements, sys, step)
end
```

## Minimal continuous-time workflow

```julia
using Random
using MonteCarloX

rng = MersenneTwister(123)
alg = Gillespie(rng)

rates = [1.0, 2.0, 0.2]  # event intensities

for _ in 1:10_000
    t, event = step!(alg, rates)
    # 1) measure at time t
    # 2) apply event update to state
    # 3) refresh rates if needed
end
```

## Where to go next

- [Core Abstractions](@ref)
- [Weights](@ref)
- [Importance Sampling Algorithms](importance_sampling_algorithms.md)
- [Continuous-Time Sampling Algorithms](continuous_time_sampling_algorithms.md)
- [Measurements](@ref)
- [Systems and model packages](@ref)
- [Worked Examples](@ref)
