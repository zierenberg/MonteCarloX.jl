# MonteCarloX.jl - Project Overview for AI Agents

## Project Mission

MonteCarloX.jl is a Julia package for Monte Carlo simulations with a focus on **clean separation of algorithms from models**. The project provides core Monte Carlo algorithms while delegating model-specific implementations to submodules or external packages.

## Core Architecture

### Design Philosophy

1. **Separation of Concerns**: Algorithms (MonteCarloX), Systems (SpinSystems, etc.), Measurements (framework)
2. **Type-Based Dispatch**: Extensible design using Julia's multiple dispatch
3. **No Systems in Core**: MonteCarloX contains ONLY algorithms; models live elsewhere
4. **Composability**: Same algorithm works with any compatible system

### Package Structure

```
MonteCarloX.jl/
├── src/                      # Core algorithms package
│   ├── measurements/         # Measurement framework
│   ├── equilibrium/          # Equilibrium algorithms (Metropolis, etc.)
│   ├── nonequilibrium/       # Non-equilibrium algorithms (Gillespie, KMC, etc.)
│   ├── event_handler.jl      # Event handling for non-equilibrium
│   ├── rng.jl                # Random number generation utilities
│   ├── utils.jl              # General utilities
│   └── MonteCarloX.jl        # Main module
├── SpinSystems/              # Spin model implementations (submodule)
├── notebooks/                # Jupyter notebooks with examples
├── examples/stash/           # Legacy examples (for reference)
├── test/                     # Unit tests
└── docs/                     # Documentation
```

## Core Abstractions

### Type Hierarchy

```julia
abstract type AbstractSystem end
abstract type AbstractLogWeight end
abstract type AbstractAlgorithm end
abstract type AbstractUpdate end
abstract type AbstractMeasurement end
```

### Key Components

#### 1. Systems (AbstractSystem)
- Represent what you simulate (Ising model, particle system, Bayesian problem)
- Live in **submodules** (SpinSystems) or **external packages**
- Provide state, observables, and update interfaces
- **Never** defined in MonteCarloX core

#### 2. Algorithms (AbstractAlgorithm)
- Represent how you sample (Metropolis, Gillespie, KMC)
- Live in **MonteCarloX core**
- Hold RNG, logweight, statistics
- Work with any compatible system

#### 3. LogWeights (AbstractLogWeight)
- Weight functions for importance sampling
- Examples: `BoltzmannLogWeight`, `MulticanonicalWeight`
- Callable objects: `lw(energy) -> log_weight`

#### 4. Measurements
- Framework for tracking observables
- Two schedules: `IntervalSchedule` (every N steps), `PreallocatedSchedule` (specific times)
- Container: `Measurements{K,S}` holds multiple named measurements

## Module Organization

### src/measurements/
Contains the measurement framework:
- `measurements.jl` - Core measurement types and schedules

### src/equilibrium/
Equilibrium Monte Carlo algorithms:
- `abstractions.jl` - Base types (AbstractSystem, AbstractLogWeight, etc.)
- `equilibrium.jl` - Metropolis and importance sampling algorithms

### src/nonequilibrium/
Non-equilibrium algorithms for stochastic processes:
- `gillespie.jl` - Gillespie algorithm (exact stochastic simulation)
- `kinetic_monte_carlo.jl` - Kinetic Monte Carlo
- `poisson_process.jl` - Poisson process simulations

### Utilities (src/ root level)
- `event_handler.jl` - Event scheduling/handling for non-equilibrium
- `rng.jl` - Random number generation utilities
- `utils.jl` - General utility functions (log_sum, binary_search, etc.)

## Submodules

### SpinSystems/
Spin model implementations:
- `Ising` - 2D Ising model
- `BlumeCapel` - Spin-1 system with crystal field
- Future: Potts, XY, Heisenberg models

### BayesianInference/ (Planned)
Bayesian inference components:
- Hierarchical models
- Inference algorithms
- Integration with Turing.jl concepts

## Usage Pattern

```julia
using MonteCarloX
using MonteCarloX.SpinSystems

# 1. Create system
rng = MersenneTwister(42)
sys = Ising([8, 8])
init!(sys, :random, rng=rng)

# 2. Setup algorithm
alg = Metropolis(rng, β=0.4)

# 3. Configure measurements
measurements = Measurements([
    :energy => energy => Float64[]
], interval=10)

# 4. Run simulation
for i in 1:10000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# 5. Analyze
mean_E = mean(measurements[:energy].data)
```

## Key Design Decisions

### Why Submodules?
- **Tight integration** with MonteCarloX
- **Clear organization** of related code
- **Single repository** for easier development
- Can become separate packages later

### Why Callable LogWeights?
```julia
struct BoltzmannLogWeight
    β::Real
end
(lw::BoltzmannLogWeight)(E) = -lw.β * E
```
- Store parameters (β) with function
- Type-safe dispatch
- Clean syntax: `lw(E)` vs `evaluate(lw, E)`

### Why Mutable Algorithms?
```julia
mutable struct Metropolis
    steps::Int
    accepted::Int
end
```
- Track statistics without global state
- Easy reset between runs
- Cleaner function signatures

## File Organization Rules

### DO's:
- ✅ Put algorithms in `src/equilibrium/` or `src/nonequilibrium/`
- ✅ Put systems in submodules (SpinSystems/) or external packages
- ✅ Put examples/tutorials in `notebooks/`
- ✅ Keep utilities at `src/` root if general-purpose
- ✅ Use descriptive module organization

### DON'Ts:
- ❌ Put system implementations in MonteCarloX core
- ❌ Mix algorithms and models in same file
- ❌ Create examples with old/incompatible API
- ❌ Add dependencies unless necessary

## Testing Strategy

- Unit tests for algorithms in `test/`
- Statistical tests for stochastic correctness
- Integration tests for system-algorithm interaction
- Performance benchmarks for critical paths

## Documentation

- `README.md` - High-level overview
- `docs/API_DESIGN.md` - Design rationale and patterns
- `docs/MIGRATION_GUIDE.md` - Old → new API transition
- `SpinSystems/README.md` - Model documentation
- `notebooks/` - Tutorial notebooks

## Development Workflow

1. **Adding New Algorithm**: Put in `src/equilibrium/` or `src/nonequilibrium/`
2. **Adding New System**: Put in appropriate submodule (SpinSystems, etc.)
3. **Adding Example**: Create notebook in `notebooks/`
4. **Breaking Change**: Update MIGRATION_GUIDE.md

## Common Patterns

### Extending with New System
```julia
# In SpinSystems or new submodule
mutable struct MySystem <: AbstractSystem
    state::Vector{Float64}
end

my_observable(sys::MySystem) = sum(sys.state)

function my_update!(sys::MySystem, alg::Metropolis)
    # propose, evaluate, accept/reject
end
```

### Extending with New Algorithm
```julia
# In src/equilibrium/ or src/nonequilibrium/
mutable struct MyAlgorithm <: AbstractAlgorithm
    rng::AbstractRNG
    # ... parameters
end

function my_step!(sys::AbstractSystem, alg::MyAlgorithm)
    # algorithm logic
end
```

## Future Directions

- **BayesianInference** submodule
- **Advanced ensembles** (multicanonical, Wang-Landau, parallel tempering)
- **More models** in SpinSystems (Potts, XY, Heisenberg)
- **Additional submodules** (ContinuousSystems, NetworkSystems, PolymerSystems)
- **Enhanced measurements** (autocorrelation, error estimation)

## Important Notes for AI Agents

1. **Never add system implementations to MonteCarloX core**
2. **Maintain clean separation**: algorithms ≠ systems ≠ measurements
3. **Use type-based dispatch** for extensibility
4. **Keep backward compatibility** when possible
5. **Document design decisions** in API_DESIGN.md
6. **Put examples in notebooks/**, not root examples/
7. **Organize by purpose**: equilibrium/, nonequilibrium/, measurements/

## Quick Reference

**Adding equilibrium algorithm**: `src/equilibrium/my_algorithm.jl`  
**Adding non-equilibrium algorithm**: `src/nonequilibrium/my_algorithm.jl`  
**Adding system**: `SpinSystems/src/my_system.jl` or new submodule  
**Adding example**: `notebooks/my_example.jl` or `.ipynb`  
**Adding utility**: `src/utils.jl` or new file at `src/`  

## Version Information

- **Current API**: Based on `notebooks/api.ipynb` design
- **Julia Version**: 1.0+
- **Key Dependencies**: Random, StatsBase, Graphs (for SpinSystems)

---

*This document serves as the canonical reference for understanding MonteCarloX.jl's architecture and design decisions. Keep it updated when making structural changes.*
