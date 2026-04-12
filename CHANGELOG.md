# Changelog

All notable changes to MonteCarloX.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-11

### Added
- Parallel tempering algorithm with support for diverse parallelization backends (threads, Distributed, MPI)
- Message-passing backend for parallel algorithms
- Multicanonical sampling with parallelization support
- Comprehensive example gallery: Bayesian inference (coin flip, house price prediction, eight schools), stochastic processes (Poisson, dimerization, Ornstein-Uhlenbeck), large deviation theory, and spin systems
- Literate.jl-based examples that auto-generate documentation and can be run interactively
- Documenter.jl-based documentation with guides and API reference

### Changed
- Refactored API around ensemble-based design (Boltzmann, Multicanonical ensembles)
- Reorganized code structure and file naming conventions
- Reorganized and verified tests (unit and some math tests)

## [0.1.0] - Initial release

### Added
- Core Monte Carlo algorithms: Metropolis, heat bath, Gillespie, importance sampling
- Kinetic Monte Carlo
- SpinSystems subpackage (Ising 2D, Blume-Capel models)
- Multicanonical sampling
- Basic measurement utilities
- Binned data structures
