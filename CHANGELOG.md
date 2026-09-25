# Changelog

All notable changes to MonteCarloX.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- General replica-exchange ladder: `ReplicaExchange(ensemble, parameters; ...)` builds one replica
  per ladder parameter from a `parameter -> ensemble` function, so the tempered quantity need not
  be a temperature. `ParallelTempering(betas)` is now the Boltzmann point of this constructor.
- Replica exchange accepts a **non-scalar reaction coordinate**, which is what lets a ladder temper
  the strength of an auxiliary Hamiltonian term (coordinate `(E₀, E₁)` for a weight
  `-β(E₀ + λE₁)`); the shared part cancels from the exchange ratio on its own.
- `advance!(sweep!, alg_or_chains, states, n)` on `AbstractAlgorithm` and `ParallelChains`: repeat
  the caller's own `sweep!(state, alg)` `n` times — one chain, or every chain in parallel with the
  repeat loop inside the parallel region (one thread barrier per call, not per sweep). Nothing in
  its signature is specific to replica exchange; the exchange stays an explicit `update!` in the
  caller's loop so the two alternating halves remain visible. What to record between rounds stays
  the caller's business: `ensemble_index(rx, r)` gives the ladder slot replica `r` currently sits on.
- External integration examples: `examples/external/sunny/sunny_interop.jl` (one Sunny model and
  one local move driving PT, Wang-Landau, REWL, and PT over Sunny's Langevin integrator) and
  `examples/external/smoqy/` (SmoQyDQMC determinant QMC with a replica-exchange ladder in a
  Hamiltonian coupling rather than a temperature).

### Removed
- The Sunny.jl benchmark comparison (`benchmarks/Sunny/`, its rows in `benchmarks.tsv` and the
  section on the spin-systems page). MonteCarloX aims to *drive* Sunny models, not race them, so a
  head-to-head speed table works against the integration it is meant to support.

### Changed
- `update!(rx, coordinates)` renamed to **`attempt_exchange!`**, pairing with the existing
  `attempt_exchange_pair!`. The generic verb said nothing about the operation and collided with the
  `update!(sys, alg)` convention the examples use for a model move; the new name also states that a
  round is a round of *attempts*, not guaranteed swaps. No deprecated forward — replica exchange is
  not released API yet.
- `index(rx, r)` renamed to **`ensemble_index(rx, r)`** — it answers *which ensemble this replica is
  currently carrying*, the counterpart of `ensemble(alg)`. Configurations never move between chains
  or ranks; ensembles do, and this index is the label that travels with them.
- **MPI pair exchange is now three-phase.** The log-ratio factorizes into two halves, each evaluable
  from one rank's own ensemble, so only the coordinates and one scalar per rank are needed to
  decide; the ensemble itself is sent only when an exchange is accepted. Previously the full
  ensemble was serialized on every attempt in both directions — 8 bytes for `BoltzmannEnsemble`, but
  `O(bins)` for a tabulated multicanonical / Wang-Landau ensemble, paid even on rejection. Both
  ranks still form the same `log_ratio` bitwise (float addition is commutative), so the accept
  decision is unchanged. Covered by a new tabulated-ensemble case in the 2-rank MPI test.
- `accept!(alg, Δarg)` no longer requires `Δarg::Real`; any coordinate the ensemble's logweight is
  linear in is accepted.
- `test/mpi/test_replica_exchange_mpi.jl`: `acceptance_rate(pt)` is collective (`MPI.Reduce`) but
  was called inside an `is_root` branch, so the non-root rank never joined and the opt-in
  `MCX_TEST_MPI=true` suite deadlocked. Hoisted out of the branch.
- The Sunny examples are consolidated into one interop page. `sunny_ising.jl`,
  `sunny_heisenberg.jl`, `sunny_pt.jl`, `sunny_wl.jl` and `sunny_rewl.jl` are removed: their
  Sunny-native-vs-MCX comparison stages are dropped (see Removed), and the remaining interop
  content is written once in `sunny_interop.jl` instead of five times. Wrapper types
  (`SunnyIsing`, `SunnyHeisenberg`) are gone with them — a Sunny `System` plus an energy
  accumulator is all MCX needs. Cuts the Sunny docs build from six Literate runs to one.

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
