# Worked Examples

This page collects example patterns from the repository and maps them to learning goals.

## A. Generic equilibrium sampling (no model package)

**Goal:** understand `Metropolis` + `Measurements` with the smallest possible state.

Pattern:

- state is a scalar `x`
- custom `logweight(x)`
- proposal `x_new = x + randn(...)`
- accept/reject with `accept!`

Start from the README “generic Metropolis” example.

## B. Ising model (with SpinSystems)

**Goal:** run a full lattice model workflow with minimal boilerplate.

Pattern:

- `sys = Ising([L, L], J=..., periodic=true)`
- `init!(sys, :random, rng=...)`
- `alg = Metropolis(rng; β=...)`
- loop `spin_flip!` and `measure!`

Start from the README “Ising” example.

## C. Continuous-time birth/death process

**Goal:** learn event-driven updates with `Gillespie`.

Pattern:

- keep state (for example population size)
- map current state to rate vector
- call `step!` or `advance!`
- update state from sampled event index

See maintained notebooks in `examples/stochastic_processes/`, especially:

- `gillespie_birth_death.ipynb`
- `gillespie_dimerization.ipynb`

## D. Bayesian inference (coin-flip)

**Goal:** use a minimal, general Bayesian pattern with Metropolis.

Pattern:

- proposal from prior: `θ_new = rand(rng, prior)`
- weight by likelihood: `loglikelihood(θ)`
- accept with `accept!(alg, θ_new, θ)`
- collect posterior samples after burn-in

See `examples/bayesian_coin_flip.ipynb`.

## E. Bayesian linear regression (house prices)

**Goal:** apply the exact same pattern to a vector parameter in a non-conjugate model.

Pattern:

- proposal from priors on `(β₀, β₁, logσ)`
- weight via Gaussian-regression `loglikelihood(θ)`
- same Metropolis accept/reject loop

See `examples/house_price_prediction.ipynb`.

## F. Poisson-process primitives

**Goal:** understand low-level event-time/event-index utilities.

Relevant API:

- `next_time`
- `next_event`
- `next`
- `step!`

See `examples/stochastic_processes/kmc_poisson.ipynb`.

## G. Generalized-ensemble workflows

**Goal:** improve exploration with adaptive/tabulated weights.

Patterns:

- `Multicanonical` + `update_weight!`
- `WangLandau` + `update_f!`
- `BinnedLogWeight` for tabulated domains

See:

- `examples/stochastic_processes/muca_OU.ipynb`
- `examples/muca_LDT_gaussian_rngs.ipynb`
- `examples/spin_systems/muca_ising2D.ipynb`

## H. Parallel generalized-ensemble example

**Goal:** replica-like histogram sharing across ranks.

See script:

- `examples/spin_systems/muca_mpi_ising2D.jl`

This demonstrates `ParallelMulticanonical` and MPI-based histogram/logweight synchronization.

## Suggested progression

1. generic Metropolis
2. Ising + measurements
3. Bayesian coin-flip (prior proposal + likelihood weight)
4. Bayesian house-price regression (same pattern in higher dimension)
5. Gillespie (step-based)
6. `advance!` callbacks
7. multicanonical / Wang-Landau
8. parallel generalized ensembles
