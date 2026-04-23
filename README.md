<img src="docs/logo/banner.png" alt="MonteCarloX.jl" width="800" />

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zierenberg.github.io/MonteCarloX.jl/dev)
[![CI: core](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI_core.yml/badge.svg)](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI_core.yml)
[![CI: examples](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI_examples.yml/badge.svg)](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI_examples.yml)
[![codecov](https://codecov.io/gh/zierenberg/MonteCarloX.jl/branch/main/graph/badge.svg?token=FUn6qFnsSN)](https://codecov.io/gh/zierenberg/MonteCarloX.jl)

MonteCarloX.jl is a modular Monte Carlo framework in Julia.
It separates the sampling algorithm from the system under study:
the user defines the system state and proposes changes; MonteCarloX provides the acceptance criterion.
Because the algorithm is independent of the model, every simulation becomes a template — replacing the system yields a new application without modifying the algorithmic loop.

Companion packages can encapsulate entire model families (e.g., `SpinSystems` for Ising and Blume-Capel models), providing system definitions, update rules, and observables out of the box.

## Example

Sample a one-dimensional Gaussian target distribution via Metropolis.
The user defines the state, the proposal, and the log-weight; MonteCarloX decides whether to accept.

```julia
using Random, MonteCarloX

rng = Xoshiro(42)

# Problem-specific: state, proposal, and target distribution
x = 0.0
propose(x) = x + 0.5 * randn(rng)
logweight(x) = -0.5 * x^2

# Algorithm (reusable across any system with a log-weight)
alg = Metropolis(rng, logweight)

# Simulation loop
for step in 1:100_000
    x_new = propose(x)
    if accept!(alg, x_new, x)    # MonteCarloX decides
        x = x_new
    end
end
```

Replace `x` with a spin configuration, a parameter vector, or any other state —
the algorithm and the loop remain the same.
Swap `Metropolis` for `Multicanonical` or `WangLandau` and only the acceptance criterion changes.

## Algorithms

- **Importance sampling**: `Metropolis`, `Glauber`, `HeatBath` — accept/reject based on a log-weight ratio.
- **Flat-histogram methods**: `Multicanonical`, `WangLandau` — iteratively adapt weights for uniform sampling over an order parameter.
- **Extended-ensemble methods**: `ParallelTempering`, `ReplicaExchange` — exchange configurations across parameters to overcome free-energy barriers.
- **Continuous-time sampling**: `Gillespie` — exact stochastic simulation via event rates.

## Application domains

The [documentation](https://zierenberg.github.io/MonteCarloX.jl/dev) includes worked examples across several domains.
Each serves as a template whose algorithmic structure is independent of the specific system.

- **Bayesian inference**: posterior sampling for coin flips, linear regression, hierarchical models.
- **Statistical mechanics**: Ising and Blume-Capel models with importance sampling, multicanonical sampling, and parallel tempering.
- **Stochastic processes**: Poisson processes, birth-death dynamics, reversible dimerization via the Gillespie algorithm.
- **Large deviation theory**: multicanonical sampling of rare fluctuations in sums of random variables and the Ornstein-Uhlenbeck process.

## Installation

```julia
import Pkg; Pkg.add("MonteCarloX")
```

Optional companion package (from repository root):

```julia
using Pkg; Pkg.develop(path="SpinSystems")
```

## Documentation

Build locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

## Testing

- Core: `julia --project -e 'using Pkg; Pkg.test()'`
- SpinSystems: `julia --project=SpinSystems -e 'using Pkg; Pkg.test()'`
