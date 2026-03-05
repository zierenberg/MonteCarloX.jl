# Importance Sampling Algorithms

Importance sampling in MonteCarloX is built around **local proposals + acceptance rules**.
The defining feature is a **discrete-step** update loop (proposal → decision → update).

While this is most commonly used for equilibrium sampling, the same machinery can be used in non-equilibrium protocols by changing parameters or target weights over steps.

## Mental model

Each step is:

1. propose a local change
2. compute a local log-ratio (or local energy difference)
3. accept/reject via the algorithm
4. update counters and measure if needed

The core API function is `accept!`.

## Metropolis

### When to use it

- default first choice for equilibrium sampling
- simple and robust

### Acceptance intuition

- always accept moves toward larger target weight
- accept less favorable moves with probability `exp(log_ratio)`

### Minimal usage

```julia
using Random
using MonteCarloX

rng = MersenneTwister(1)
logweight(x) = -0.5 * x^2
alg = Metropolis(rng, logweight)

x = 0.0
for _ in 1:20_000
    x_new = x + randn(alg.rng)
    x = accept!(alg, x_new, x) ? x_new : x
end

println(acceptance_rate(alg))
```

## Glauber

Same proposal style as Metropolis, but uses logistic acceptance.
Useful when that acceptance rule is the natural one for your dynamics/modeling convention.

## HeatBath

Draws from local conditional probabilities instead of accept/reject.
For Ising-like models this often means directly sampling local spin values from conditional weights.

## Generalized ensembles

These methods adapt or use non-canonical weights to improve exploration.

### Multicanonical

- keeps a histogram of visited bins
- updates tabulated log-weights from histogram information
- useful for broad energy exploration / barrier crossing

```julia
using Random
using MonteCarloX

lw = BinnedLogWeight(-20:2:20, 0.0)
alg = Multicanonical(MersenneTwister(2), lw)

set_logweight!(alg, -10:2:10, x -> 0.0)
# run your update loop with accept!(alg, x_new, x_old)
# then call update_weight!(alg)
```

### Wang-Landau

- updates log-density-of-states estimate at visited bins
- progressively refines modification factor (`logf` via `update_f!`)

```julia
using Random
using MonteCarloX

lw = BinnedLogWeight(-20:2:20, 0.0)
alg = WangLandau(MersenneTwister(3), lw; logf=1.0)

# in your loop: accept!(alg, x_new, x_old)
# between stages: update_f!(alg)
```

## Choosing quickly

- Start with `Metropolis` for standard equilibrium sampling.
- Use `HeatBath` when conditional local probabilities are natural and cheap.
- Use `Multicanonical`/`WangLandau` when canonical sampling gets stuck or explores too narrowly.

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
set_logweight!
update_weight!
update_f!
```