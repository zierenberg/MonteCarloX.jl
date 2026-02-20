# MonteCarloX API Design

## Overview

This document describes the new API design implemented in this branch, based on `examples/api.ipynb`. The design prioritizes **separation of concerns** and **composability**.

## Design Principles

### 1. Separation of Concerns

The API cleanly separates three orthogonal concepts:

- **System**: What you simulate (Ising model, particle system, Bayesian inference problem)
- **Algorithm**: How you sample (Metropolis, Gillespie, multicanonical)
- **Measurement**: What you observe (energy, magnetization, custom observables)

This separation allows:
- Algorithms to work with any compatible system
- Systems to be used with any compatible algorithm
- Measurements to be configured independently

### 2. Type-Based Dispatch

The API leverages Julia's multiple dispatch:

```julia
abstract type AbstractSystem end
abstract type AbstractLogWeight end
abstract type AbstractAlgorithm end

# Different systems
struct Ising <: AbstractSystem ... end
struct BlumeCapel <: AbstractSystem ... end

# Different weights
struct BoltzmannLogWeight <: AbstractLogWeight ... end

# Different algorithms
struct Metropolis <: AbstractAlgorithm ... end
```

This enables:
- Extensibility without modifying core code
- Type-safe interfaces
- Compiler optimizations

### 3. No Systems in Core

**Critical design decision**: MonteCarloX core contains NO system implementations.

Systems live in:
- **Submodules**: SpinSystems.jl for spin models
- **External packages**: User packages for custom systems
- **Examples**: Demonstration systems

This ensures:
- MonteCarloX stays focused on algorithms
- Clear organizational structure
- Easy addition of new models without cluttering core

## Core Abstractions

### AbstractSystem

Base type for all systems. A system must provide:
- State (spins, particles, latent variables, etc.)
- Observables (energy, magnetization, etc.)
- Interface for updates

```julia
abstract type AbstractSystem end

# Example implementation
mutable struct Ising <: AbstractSystem
    spins::Vector{Int8}
    # ... other fields
end

# Required interface
energy(sys::Ising) = ...
```

### AbstractLogWeight

Weight functions for importance sampling. Common examples:

```julia
struct BoltzmannLogWeight <: AbstractLogWeight
    β::Real  # inverse temperature
end

(lw::BoltzmannLogWeight)(E) = -lw.β * E

# Future: MulticanonicalWeight, WangLandauWeight, etc.
```

### AbstractAlgorithm

Base type for Monte Carlo algorithms. Typically contains:
- RNG for reproducibility
- LogWeight function
- Statistics (steps, acceptance, etc.)

```julia
mutable struct Metropolis <: AbstractAlgorithm
    rng::AbstractRNG
    logweight::Base.Callable
    steps::Int
    accepted::Int
end
```

### Update Functions

Updates coordinate system and algorithm:

```julia
function spin_flip!(sys::Ising, alg::Metropolis)
    # 1. Propose change
    i = rand(alg.rng, 1:length(sys.spins))
    ΔE = delta_energy(sys, i)
    
    # 2. Evaluate acceptance
    log_ratio = alg.logweight(ΔE)
    
    # 3. Accept/reject
    if accept!(alg, log_ratio)
        modify!(sys, i, ΔE)
    end
end
```

Note: Updates can be:
- Model-specific: `spin_flip!(::Ising, ::AbstractAlgorithm)`
- Algorithm-specific: `cluster_update!(::AbstractSpinSystem, ::Wolff)`
- General: `metropolis_step!(::AbstractSystem, ::Metropolis)`

## Measurement Framework

Inspired by the excellent design in `api.ipynb`.

### Basic Structure

```julia
# Define observable and container
measurement = Measurement(
    energy,           # observable function
    Float64[]         # data container
)

# Measure
measure!(measurement, sys)  # pushes energy(sys) to data
```

### Measurement Schedules

Two types of schedules:

1. **IntervalSchedule**: Measure every N steps (indefinite simulations)
```julia
measurements = Measurements([
    :energy => energy => Float64[]
], interval=10)

for i in 1:100000
    update!(sys, alg)
    measure!(measurements, sys, i)  # measures at i=10,20,30,...
end
```

2. **PreallocatedSchedule**: Measure at specific times (finite simulations)
```julia
times = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
measurements = Measurements([
    :observable => obs => Float64[]
], times)

measure!(measurements, sys, t)  # automatically handles all t
```

### Multiple Measurements

```julia
measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[],
    :specific_heat => (sys -> energy(sys)^2) => Float64[]
], interval=10)

# Access results
E_data = measurements[:energy].data
M_data = measurements[:magnetization].data
```

## Example: Complete Simulation

```julia
using Random
using MonteCarloX
using MonteCarloX.SpinSystems

# 1. Create system
rng = MersenneTwister(42)
sys = Ising([16, 16], J=1, periodic=true)
init!(sys, :random, rng=rng)

# 2. Setup algorithm
alg = Metropolis(rng, β=0.44)

# 3. Configure measurements
measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=100)

# 4. Thermalize
N = 16^2
for i in 1:N*1000
    spin_flip!(sys, alg)
end

# 5. Production
reset_statistics!(alg)
for i in 1:N*100000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# 6. Analyze
using StatsBase
println("⟨E⟩ = ", mean(measurements[:energy].data) / N)
println("Acceptance = ", acceptance_rate(alg))
```

## Extending the API

### Adding a New System

Create a new type inheriting from AbstractSystem:

```julia
mutable struct MySystem <: AbstractSystem
    state::Vector{Float64}
    # ... other fields
end

# Implement observables
my_observable(sys::MySystem) = sum(sys.state)

# Implement update
function my_update!(sys::MySystem, alg::Metropolis)
    # propose, evaluate, accept/reject
end
```

### Adding a New Algorithm

Create a new algorithm type:

```julia
mutable struct HeatBath <: AbstractImportanceSampling
    rng::AbstractRNG
    logweight::Base.Callable
    β::Real
    steps::Int
end

# Implement update for your algorithm
function heatbath_step!(sys::Ising, alg::HeatBath)
    # Heat bath specific logic
end
```

### Adding a New Weight Function

```julia
struct MulticanonicalWeight <: AbstractLogWeight
    log_dos::Vector{Float64}  # Density of states
    energies::Vector{Float64}
end

function (lw::MulticanonicalWeight)(E)
    idx = searchsorted(lw.energies, E)
    return -lw.log_dos[idx]
end
```

## Comparison with Legacy API

### Legacy (Old)
```julia
# Systems defined in core
# Limited type structure
# Measurements ad-hoc
accept(Metropolis(), rng, β, ΔE)
```

### New API
```julia
# Systems in submodules
# Rich type hierarchy
# Structured measurement framework
alg = Metropolis(rng, β=β)
if accept!(alg, log_ratio)
    modify!(sys, ...)
end
```

## Future Directions

### Planned Extensions

1. **Advanced ensemble methods**
   - Multicanonical
   - Wang-Landau
   - Parallel tempering
   - Population annealing

2. **BayesianInference submodule**
   - Hierarchical models
   - Inference algorithms
   - Integration with Turing.jl concepts

3. **More submodules**
   - ContinuousSystems.jl
   - NetworkSystems.jl
   - PolymerSystems.jl

4. **Enhanced measurements**
   - Autocorrelation analysis
   - Error estimation
   - Online statistics

## Design Decisions Rationale

### Why submodules instead of separate packages?

For now, submodules provide:
- Tight integration with MonteCarloX
- Easier development iteration
- Single repository for related code

Future: May become separate packages with proper dependencies.

### Why callable objects for LogWeight?

```julia
struct BoltzmannLogWeight
    β::Real
end
(lw::BoltzmannLogWeight)(E) = -lw.β * E
```

Benefits:
- Store parameters (β) with function
- Type-safe dispatch
- Clean syntax: `lw(E)` instead of `evaluate(lw, E)`

### Why mutable algorithm structs?

```julia
mutable struct Metropolis
    steps::Int
    accepted::Int
end
```

Allows:
- Tracking statistics without global state
- Easy reset between runs
- Cleaner function signatures

## Conclusion

This API design provides:
- ✓ Clean separation of concerns
- ✓ Extensibility through types
- ✓ No systems in core package
- ✓ Composable algorithms and models
- ✓ Robust measurement framework

Based on the proven design from `examples/api.ipynb` and ready for interactive refinement.
