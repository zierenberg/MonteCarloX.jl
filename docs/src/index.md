# MonteCarloX.jl

MonteCarloX is a composable framework for Monte Carlo simulations in Julia.
It supports equilibrium and non-equilibrium workflows while keeping model code
separate from algorithmic code.

## Philosophy

The core design principle is separation of concerns:

- **System**: state and model-specific operations
- **Weight/rates**: probability structure (equilibrium) or event intensities (dynamics)
- **Algorithm**: how proposals/events are sampled
- **Update**: how accepted events modify state
- **Measurement**: what is recorded and when

This allows reusable algorithm templates where systems can be swapped without
rewriting the simulation driver.

## Documentation roadmap

If you are new to the package, read in this order:

1. Framework
2. Core Abstractions
3. Weights
4. Importance Sampling or Continuous-Time Sampling Algorithms
5. Measurements
6. Systems
7. Worked Examples

## Scope

MonteCarloX contains the algorithmic and infrastructure core.
Concrete model families are expected to live in companion packages/modules,
such as `SpinSystems`.

## Contributing

The API is actively refined. Contributions are welcome, especially around
examples, docs coverage, and cross-domain model integrations.

