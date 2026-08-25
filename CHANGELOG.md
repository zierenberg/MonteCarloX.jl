# Changelog

All notable changes to MonteCarloX.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-25

### Added
- benchmarks
- inference use cases (still experimental)
- reweighting
- extension to include Distributions.jl
- on-compile checks that could break usage: e.g. if using dx API for Metropolis the logweight has to be linear

### Changed
- documentation incl. real examples that are precomputed
- cleanup in algorithms
- organization of system modules (long-term goal is to have them external)
- organization of algorithms, ensembles, etc
- accept API clear and concise
- restructured event handlers (kMC)

## [0.2.0] - 2026-06-08

### Added
- Parallel tempering algorithm with support for diverse parallelization backends (threads, Distributed, MPI)
- Message-passing backend for parallel algorithms
- Multicanonical sampling with parallelization support
- Advanced multicanonical helpers (weight updates, convergence diagnostics)
- Binning utilities for histogram-based analysis
- `logweight` helper functions for common ensemble compositions
- Comprehensive example gallery: Bayesian inference (coin flip, house price prediction, eight schools), stochastic processes (Poisson, dimerization, Ornstein-Uhlenbeck), large deviation theory, and spin systems
- Literate.jl-based examples that auto-generate documentation and can be run interactively
- Documenter.jl-based documentation with guides and API reference
- Companion package scaffolding for `MCXSoftMatter` (off-lattice particles and bead-spring polymers) and `MCXLatticeMatter` (lattice polymers with translate/slither/pivot/double-bridge moves)

### Changed
- Refactored API around ensemble-based design (Boltzmann, Multicanonical ensembles)
- Reorganized code structure and file naming conventions
- Reorganized and verified tests (unit and some math tests)
- Corrected energy caching along the importance-sampling path
- Tightened argument validation (e.g., error on non-positive step counts)
- Tightened `[compat]` bounds and set `julia = "1.10"` in preparation for registration

## [0.1.0] - Initial release

### Added
- Core Monte Carlo algorithms: Metropolis, heat bath, Gillespie, importance sampling
- Kinetic Monte Carlo
- MCXSpins subpackage (Ising 2D, Blume-Capel models)
- Multicanonical sampling
- Basic measurement utilities
- Binned data structures
