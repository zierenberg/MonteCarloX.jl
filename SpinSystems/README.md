# SpinSystems.jl

A submodule of MonteCarloX.jl providing spin system implementations.

## Overview

SpinSystems.jl provides concrete implementations of spin models for use with MonteCarloX.jl's Monte Carlo algorithms. This separation allows MonteCarloX to focus on algorithms while SpinSystems handles model-specific details.

## Implemented Models

### Ising Model

The classic ferromagnetic Ising model with Hamiltonian:
```
H = -J ∑_{<i,j>} sᵢsⱼ
```

**Features:**
- Efficient bookkeeping of energy and magnetization
- Supports arbitrary graphs via `Graphs.jl`
- Convenience constructor for hypercubic lattices
- Optimized spin flip updates

**Example usage:**
```julia
using MonteCarloX
using MonteCarloX.SpinSystems

# Create 8×8 Ising model
sys = Ising([8, 8], J=1, periodic=true)
init!(sys, :random, rng=rng)

# Setup Metropolis algorithm
alg = Metropolis(rng, β=0.4)

# Perform updates
for i in 1:1000
    spin_flip!(sys, alg)
end
```

### Blume-Capel Model

Spin-1 system with crystal field, Hamiltonian:
```
H = -J ∑_{<i,j>} sᵢsⱼ + D ∑ᵢ sᵢ²
```

where spins can take values {-1, 0, +1}.

**Features:**
- Three-state spin system
- Crystal field parameter D
- Similar interface to Ising model

**Example usage:**
```julia
using MonteCarloX
using MonteCarloX.SpinSystems

# Create BlumeCapel model
sys = BlumeCapel([8, 8], J=1, D=0.5, periodic=true)
init!(sys, :random, rng=rng)

# Use with any MonteCarloX algorithm
alg = Metropolis(rng, β=0.4)
for i in 1:1000
    spin_flip!(sys, alg)
end
```

## API

All spin systems inherit from `AbstractSpinSystem` and provide:

### Constructors
- `Model(graph::SimpleGraph, ...)`: Create from arbitrary graph
- `Model(dims::Vector{Int}, ...)`: Create hypercubic lattice

### Initialization
- `init!(sys, :up)`: All spins up
- `init!(sys, :down)`: All spins down  
- `init!(sys, :zero)`: All spins zero (BlumeCapel only)
- `init!(sys, :random, rng=rng)`: Random configuration

### Observables
- `energy(sys)`: Total energy
- `magnetization(sys)`: Absolute magnetization
- `delta_energy(sys, i, ...)`: Energy change for proposed update

### Updates
- `spin_flip!(sys, alg)`: Single spin update with algorithm
- `modify!(sys, ...)`: Apply accepted update

## Integration with MonteCarloX

SpinSystems is designed to work seamlessly with MonteCarloX's algorithms:

```julia
using Random
using StatsBase
using MonteCarloX
using MonteCarloX.SpinSystems

rng = MersenneTwister(42)
sys = Ising([8, 8])
init!(sys, :random, rng=rng)

alg = Metropolis(rng, β=0.4)

measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)

# Simulation loop
for i in 1:10000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# Analysis
println("Acceptance rate: ", acceptance_rate(alg))
println("Average energy: ", mean(measurements[:energy].data))
```

## Design Philosophy

SpinSystems follows the MonteCarloX philosophy of separation of concerns:
- **System**: Holds state and provides observables
- **Algorithm**: Implements sampling strategy
- **Update**: Coordinates system and algorithm

This allows:
- Easy swapping of systems while keeping the same algorithm
- Reusable algorithms across different models
- Clear, maintainable code structure

## Future Extensions

Planned additions:
- Potts model
- XY model
- Heisenberg model
- Optimized cluster updates (Wolff, Swendsen-Wang)
- Support for inhomogeneous systems
