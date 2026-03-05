# Systems and model packages

MonteCarloX keeps systems external on purpose.
The framework provides algorithms; companion packages provide concrete models.

## Why this split helps

- algorithm implementations stay generic
- model packages can evolve independently
- the same sampler can be reused across many domains

## Example: `SpinSystems` with Ising

```julia
using Random
using MonteCarloX
using SpinSystems

rng = MersenneTwister(123)
sys = Ising([16, 16], J=1.0, periodic=true)
init!(sys, :random, rng=rng)

alg = Metropolis(rng; β=0.44)

for _ in 1:100_000
    spin_flip!(sys, alg)
end

println("E = ", energy(sys))
println("|M| = ", magnetization(sys))
```

## What a custom model package should implement

At minimum:

1. a concrete system type (`AbstractSystem` subtype)
2. observable functions for analysis
3. update methods that call MonteCarloX primitives (`accept!`, etc.)
4. initialization utilities (`init!` pattern)

This is usually enough to plug your model into measurement and algorithm workflows immediately.
