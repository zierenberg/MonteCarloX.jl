# Systems and Model Packages
MonteCarloX keeps systems external on purpose.
The framework provides algorithms; companion packages provide concrete models.

## Why this split helps
- algorithm implementations stay generic
- model packages can evolve independently
- the same sampler can be reused across many domains

## Example: `MCXSpins` with Ising
```julia
using Random
using MonteCarloX
using MCXSpins

rng = MersenneTwister(123)
sys = IsingSystem([16, 16]; J=1.0)          # −J Σ_{<ij>} σσ on a periodic 16×16 lattice
init!(sys, :random, rng=rng)
alg = MetropolisAlgorithm(rng; β=0.44)

for _ in 1:100_000
    spin_flip!(sys, alg)
end

println("E  = ", energy(sys))
println("|M| = ", magnetization(sys))
```

In `MCXSpins` a system is a spin type plus a tuple of interaction terms — the one-liner
above is sugar for

```julia
sys = SpinSystem(Spin(1//2), (PairInteraction(1.0, partners),))
```

so new models are composed from existing terms (`PairInteraction`,
`PairInteractionMatrix`, `ExternalField`, `CrystalField`, …) rather than written from
scratch.

## What a custom model package should implement
At minimum:

1. a concrete system type (`AbstractSystem` subtype)
2. observable functions for analysis
3. update methods that call MonteCarloX primitives (`accept!`, etc.)
4. initialization utilities (`init!` pattern)

This is usually enough to plug your model into measurement and algorithm workflows immediately.