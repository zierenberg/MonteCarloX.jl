# Framework: how to think in MonteCarloX

MonteCarloX is not a monolithic simulator.
It is a toolkit of small building blocks that you compose explicitly.

## The simulation recipe

Every workflow needs these 3 parts:

1. **System** (`AbstractSystem`): your current state
2. **Weight/rates** (`AbstractLogWeight` or event rates): your target physics/statistics
3. **Algorithm** (`AbstractAlgorithm`): how transitions are sampled

Optional convenience layer:

4. **Measurements** (`Measurement` / `Measurements`): helper containers and schedules for recording observables

This gives you a clean separation: model logic lives in systems, sampling logic lives in algorithms.

## Two paradigms, one interface style

### 1) Importance sampling (discrete-step updates)

Primary characteristic: the simulation advances in discrete proposal/update steps.

This is often used for equilibrium sampling, but it can also be used for non-equilibrium protocols (for example driven schedules in parameters or weights).

Loop structure:

1. Propose local change
2. Compute local log-ratio / energy difference
3. Accept/reject
4. (Optional) record observables

### 2) Continuous-time sampling (dynamics)

Goal: evolve a stochastic process in physical/simulation time.

Loop structure:

1. Provide rates (or event handler)
2. Sample `(dt, event)`
3. Advance time by `dt`
4. (Optional) record observables at time stamps that were passed
5. Apply event update

## Minimal equilibrium sketch

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
logweight(x) = -0.5 * x^2
alg = Metropolis(rng, logweight)

x = 0.0
for _ in 1:10_000
    x_new = x + randn(alg.rng)
    x = accept!(alg, x_new, x) ? x_new : x
end
```

## Minimal continuous-time sketch

```julia
using Random
using MonteCarloX

alg = Gillespie(MersenneTwister(2))
rates = [0.2, 0.8]

for _ in 1:10_000
    t, event = step!(alg, rates)
    # update your state using `event`
end
```

## Next pages

- Build Your Own System
- Core Abstractions
- Importance Sampling Algorithms
- Continuous-Time Sampling Algorithms
- Measurements
- Worked Examples
