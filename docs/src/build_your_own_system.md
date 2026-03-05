# Build Your Own System

This is the most practical way to start with MonteCarloX.

## Minimal goal

To run a custom model with MonteCarloX, you typically need:

1. A state type (your system)
2. A way to evaluate local quantities used by your update
3. An update function that calls MonteCarloX algorithm primitives (`accept!` or `step!`)

You do **not** need to implement the full framework at once.

## Example: minimal scalar model (`energy(x)`) with Metropolis

```julia
using Random
using MonteCarloX

energy(x) = 0.5 * x^2
delta_energy(x, x_new) = energy(x_new) - energy(x)

function local_update(x::Float64, alg::Metropolis)
    x_new = x + randn(alg.rng)
    ΔE = delta_energy(x, x_new)
    if accept!(alg, ΔE)
        return x_new
    end
    return x
end

rng = MersenneTwister(42)
alg = Metropolis(rng; β=1.0)  # uses BoltzmannLogWeight(β)
x = 0.0

for _ in 1:10_000
    x = local_update(x, alg)
end
```

This is the smallest possible pattern: define `energy(x)`, derive `ΔE`, and let a framework algorithm (`Metropolis`) handle acceptance.

## Checklist for importance-sampling systems

- Define your system state type.
- Define local proposal/update logic.
- Compute local old/new quantity (or delta) needed by `accept!`.
- Apply state changes only after acceptance.

For many models, this is enough.

## Example: birth-death system with Gillespie

```julia
using Random
using MonteCarloX

mutable struct BirthDeathState <: AbstractSystem
    n::Int
    λ::Float64
    μ::Float64
end

function rates(sys::BirthDeathState, t)
    birth = sys.λ * sys.n
    death = sys.μ * sys.n
    return [birth, death]
end

function apply_event!(sys::BirthDeathState, event)
    if event == 1
        sys.n += 1
    elseif event == 2
        sys.n = max(0, sys.n - 1)
    end
    return nothing
end

alg = Gillespie(MersenneTwister(7))
sys = BirthDeathState(25, 0.2, 0.1)

for _ in 1:10_000
    t, event = step!(alg, rates(sys, alg.time))
    event === nothing && break
    apply_event!(sys, event)
end
```

Same system with callback-style integration:

```julia
alg = Gillespie(MersenneTwister(8))
sys = BirthDeathState(25, 0.2, 0.1)

advance!(
    alg,
    sys,
    100.0;
    rates=(state, t) -> rates(state, t),
    update!=(state, event, t) -> apply_event!(state, event),
)
```

## Checklist for continuous-time systems

- Define state type.
- Define `rates(state, t)` that returns event rates.
- Define event-application logic for sampled event indices.
- Use `step!` for manual loops or `advance!` for callback-based loops.

## Where `Measurements` fit

`Measurements` are optional helpers.
You can record observables manually in vectors, or use `Measurement`/`Measurements` for scheduling convenience.

## Real model reference

For a production-ready system implementation, inspect `SpinSystems` (for example Ising), where model logic is separated cleanly from MonteCarloX algorithms.