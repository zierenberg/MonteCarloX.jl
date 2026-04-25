# Build Your Own System

MonteCarloX provides **algorithms**, not complete simulations.
Your job is to define the **system state** and **local update rules**.
The framework handles acceptance, event scheduling, and sampling infrastructure.

## Minimal requirements

To use MonteCarloX algorithms, you need:

1. **System state type** — your model's configuration
2. **Local update function** — proposes changes and uses `accept!` or `step!`
3. **Quantities needed by the algorithm** — energy difference, event rates, etc.

You do *not* need to implement the full framework.
Start simple and add advanced features incrementally.

---

## Discrete-state sampling (Metropolis, Glauber, HeatBath)

### Core pattern

```julia
using MonteCarloX

# 1. Define system state
mutable struct MySystem
    # state variables
end

# 2. Compute energy (or log-weight)
energy(sys::MySystem, i) = # ... compute (local) total energy

# 3. Local update proposes change and uses accept!
function update!(sys::MySystem, alg::Metropolis; delta=0.1)
    i = rand(alg.rng, 1:length(sys))
    # backup old state
    tmp = sys[i]
    energy_old = energy(sys, i)
    # local change
    sys[i] += delta * (2 * rand(alg.rng) - 1)
    energy_new = energy(sys, i)
    if !accept!(alg, energy_new, energy_old)
        sys[i] = tmp
    end
end

# 4. Run the simulation
alg = Metropolis(Xoshiro(42); β=1.0)
sys = MySystem(...)
for step in 1:10_000
    update!(sys, alg)
end
```

### Checklist
- Define your system state type.
- Define the total energy (or log-weight) function.
- In your update function: propose → compute energies → `accept!(alg, E_new, E_old)` → apply if accepted / revert if rejected.
- Pass the RNG from `alg.rng` to random operations.

---

## Continuous-time sampling (Gillespie)

### Core pattern

```julia
using MonteCarloX

mutable struct MyProcess <: AbstractSystem
    # state variables
end

# 1. Provide event rates (must return a vector)
function MonteCarloX.event_source(sys::MyProcess, t)
    # compute rates for each possible event
    return [rate1, rate2, ...]
end

# 2. Apply event (modify state in-place)
function MonteCarloX.modify!(sys::MyProcess, event::Int, t)
    # modify state based on event index
    if event == 1
        # birth event
    elseif event == 2
        # death event
    end
    return nothing
end

# 3. Run with step! or advance!
alg = Gillespie(Xoshiro(42))
sys = MyProcess(...)
T = 100.0

# Manual loop
for _ in 1:10_000
    t, event = step!(alg, event_source(sys, alg.time))
    event === nothing && break
    modify!(sys, event, t)
end

# Or use advance! with callbacks
advance!(alg, sys, T;
    rates=(state, t) -> event_source(state, t),
    update!=(state, event, t) -> modify!(state, event, t),
)
```

### Checklist
- Define state type (subtype `AbstractSystem`).
- Implement `event_source(state, t)` returning event rate vector.
- Implement `modify!(state, event, t)` to apply state changes.
- Use `step!` for manual control or `advance!` for callback-based integration.

---

## Measuring observables

`Measurements` is an optional helper for scheduling observations:

```julia
using MonteCarloX

# Define what to measure and when
# The second argument is either:
#   - An interval (e.g., 1:10 means measure every 10 steps)
#   - A vector of specific times/steps (e.g., collect(0:10:100) for steps 0,10,20,...)
measurements = Measurements([
    :energy => (sys -> energy(sys)) => Float64[],
    :magnetization => (sys -> magnetization(sys)) => Float64[],
], interval=10)

# At each measurement opportunity:
measure!(measurements, sys, step_index)

# Access results:
energy_data = measurements[:energy].data
```

You can measure manually with vectors if you prefer — `Measurements` just adds convenience.

---

## Key points

- **Algorithms are model-agnostic**: the same `Metropolis` works for spin systems, Bayesian posteriors, or custom models.
- **State management is yours**: MonteCarloX never modifies your system directly; updates happen in your code or mediated by other packages (cf. SpinSystems).
- **RNG is passed from algorithm**: always use `alg.rng` for randomness.
- **Checkpointing**: serialize (e.g. `sys` and `alg`) for restart capability.
- **Parallel extensions build on the same interface**: `ParallelTempering`, `ParallelMulticanonical`, etc.

See the `SpinSystems` package as example for system implementations.
