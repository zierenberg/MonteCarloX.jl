# MonteCarloX

[![Dev](https://img.shields.io/badge/docs-stable-blue.svg)](https://zierenberg.github.io/MonteCarloX.jl/dev)
[![CI Tests](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/zierenberg/MonteCarloX.jl/branch/main/graph/badge.svg?token=FUn6qFnsSN)](https://codecov.io/gh/zierenberg/MonteCarloX.jl)

MonteCarloX is an open-source project to implement basic and advanced Monte Carlo algorithms for general use. The project focuses on the programming language [Julia](https://julialang.org/), which was specifically developed for scientific computing.

## New API Design (This Branch)

This branch implements a **completely redesigned API** based on `examples/api.ipynb` that cleanly separates concerns:

- **MonteCarloX**: Core Monte Carlo algorithms (equilibrium & non-equilibrium), measurement framework
- **SpinSystems**: Spin models (Ising, Blume-Capel, etc.) as a submodule
- **BayesianInference**: Bayesian inference components (planned)

### Key Features

✓ **Clean separation**: Algorithms live in MonteCarloX, models live in submodules  
✓ **Flexible measurement framework**: Interval-based or preallocated schedules  
✓ **Reusable algorithms**: Same Metropolis works for Ising, Blume-Capel, or custom systems  
✓ **No system definitions in core**: MonteCarloX contains only algorithms  
✓ **Type-based dispatch**: Clean, extensible design using Julia's type system  
✓ **Testing boundaries**: Core tests live in `test/`; model tests live in `SpinSystems/test/`  

### Quick Start

```julia
using Random
using MonteCarloX
using SpinSystems

# Create an Ising model
rng = MersenneTwister(42)
sys = Ising([8, 8], J=1, periodic=true)
init!(sys, :random, rng=rng)

# Setup Metropolis algorithm
alg = Metropolis(rng, β=0.4)

# Configure measurements
measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)

# Run simulation
for i in 1:10000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# Analyze results
println("Acceptance: ", acceptance_rate(alg))
println("⟨E⟩ = ", mean(measurements[:energy].data))
```

### Core Abstractions

The new API is built around clean abstractions:

- `AbstractSystem`: Base type for all systems (holds state, provides observables)
- `AbstractLogWeight`: Weight functions (Boltzmann, multicanonical, etc.)
- `AbstractAlgorithm`: Monte Carlo algorithms (Metropolis, Gillespie, etc.)
- `AbstractUpdate`: Update methods coordinating system and algorithm
- `AbstractMeasurement`: Measurement infrastructure

### Package Structure

```
MonteCarloX.jl/
├── src/
│   ├── abstractions.jl        # Core types (AbstractSystem, AbstractAlgorithm, etc.)
│   ├── measurements/          # Measurement framework
│   ├── algorithms/            # All Monte Carlo algorithms
│   │   ├── event_handler.jl   # Event handling for KMC
│   │   ├── importance_sampling.jl  # Metropolis (simplest importance sampling)
│   │   ├── kinetic_monte_carlo.jl  # Gillespie (simplest KMC) and general KMC
│   │   ├── multicanonical.jl       # Multicanonical + Wang-Landau generalized ensembles
│   │   ├── parallel_tempering.jl   # (Placeholder) Replica exchange
│   │   └── population_annealing.jl # (Placeholder) Population annealing
│   ├── rng.jl                 # Random number utilities
│   ├── utils.jl               # General utilities
│   └── MonteCarloX.jl         # Main module
├── SpinSystems/               # Submodule for spin models
│   ├── src/
│   │   ├── ising.jl           # Ising model
│   │   ├── blume_capel.jl     # Blume-Capel model
│   │   └── ...
│   └── README.md
├── notebooks/                 # Jupyter notebooks with examples
│   ├── api.ipynb              # API development notebook
│   ├── simple_ising.ipynb     # Equilibrium: Ising + Metropolis
│   ├── birth_death_process.ipynb     # Non-equilibrium: branching + mean-field dynamics
│   └── poisson_kmc.ipynb       # Poisson processes via kMC primitives
├── SpinSystems/test/          # Model-level tests
├── examples/stash/            # Legacy examples (for reference)
└── docs/                      # Documentation
```

### Examples

See the notebooks in `notebooks/` for complete demonstrations:
- `api.ipynb`: end-to-end API walkthrough
- `simple_ising.ipynb`: equilibrium Ising + Metropolis
- `birth_death_process.ipynb`: branching and mean-field non-equilibrium dynamics
- `poisson_kmc.ipynb`: Poisson processes using kinetic Monte Carlo primitives

## Goal

Since Monte Carlo algorithms are often tailored to specific problems, we break them down into small basic functions that can be applied independent of the underlying models. We thereby **separate the algorithmic part from the model part**. 

MonteCarloX contains only the core algorithmic components. Models are provided by submodules (SpinSystems) or external packages. Different from other simulation packages, the goal of MonteCarloX is **not** to hide the final simulation under simple black-box function calls, but to foster the construction of clean **template simulations** where algorithms and models can be easily swapped.

## Testing

- **Core (MonteCarloX)**: Run from the repository root: `julia --project -e 'using Pkg; Pkg.test()'`
- **Models (SpinSystems)**: Run from the SpinSystems directory: `julia --project=SpinSystems -e 'using Pkg; Pkg.test()'`

## Documentation

Build the docs locally and open the generated HTML:

`julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'`

The output is written to `docs/build/index.html`.

Core tests cover algorithms, log weights, measurements, RNG utilities, and event handling. Model behavior (Ising, Blume-Capel, etc.) is validated in `SpinSystems/test/` to keep responsibilities separated.

## Roadmap

This branch is under active development. Current status:

- [x] Core abstractions (AbstractSystem, AbstractLogWeight, etc.)
- [x] Measurement framework (interval & preallocated schedules)
- [x] Equilibrium algorithms (Metropolis with Boltzmann weight)
- [x] Generalized ensembles (Multicanonical, Wang-Landau scaffolds)
- [x] SpinSystems submodule (Ising, Blume-Capel)
- [x] Working examples demonstrating the new API
- [ ] BayesianInference submodule
- [ ] Migration of existing examples to new API
- [ ] Extended documentation
- [ ] Comprehensive tests for new API
- [ ] Advanced ensemble methods (multicanonical, etc.)

## Contribute

MonteCarloX employs continuous integration with unit tests. Right now, we are developing a stable API that fits the needs of a variety of advanced algorithms. The API may still change as we refine the design. Once we have a stable version, MonteCarloX is intended as a community project.

## Related Packages

- [Carlo.jl](https://github.com/lukas-weber/Carlo.jl): Another approach to Monte Carlo simulations in Julia
