# Systems and model packages

MonteCarloX intentionally does not hard-code specific physical models.
Instead, models live in dedicated packages/modules and implement system-side methods.

## Example: `SpinSystems`

`SpinSystems` provides concrete Ising/Blume-Capel systems and update methods
compatible with MonteCarloX algorithms.

```julia
using Random
using MonteCarloX
using SpinSystems

rng = MersenneTwister(123)
sys = Ising([32, 32], J=1, periodic=true)
init!(sys, :random, rng=rng)

alg = Metropolis(rng; β=0.44)
for _ in 1:10_000
    spin_flip!(sys, alg)
end

println(energy(sys), magnetization(sys))
```

## What a custom system should provide

At minimum, your model package should expose:

- state representation (your concrete subtype of `AbstractSystem`)
- observables needed by your analysis (`energy`, order parameters, ...)
- update hooks compatible with target algorithm(s)
- initialization and reproducibility controls

This design keeps MonteCarloX reusable while allowing domain-specific model code
to evolve independently.
