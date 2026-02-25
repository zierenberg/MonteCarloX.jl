# Worked Examples

This page maps common goals to runnable examples.

## 1) Equilibrium Ising with Metropolis

**Goal:** sample canonical equilibrium and measure energy/magnetization.

- See notebook: `notebooks/simple_ising.ipynb`
- Main pieces: `Ising` (system) + `BoltzmannLogWeight` via `Metropolis(β=...)` + `Measurements`

## 2) Branching / birth-death process

**Goal:** simulate non-equilibrium stochastic population dynamics.

- See notebook: `notebooks/birth_death_process.ipynb`
- Main pieces: state variables + rates + `Gillespie` + time-based measurements

## 3) Poisson process with kinetic Monte Carlo primitives

**Goal:** simulate homogeneous and inhomogeneous Poisson processes.

- See notebook: `notebooks/poisson_kmc.ipynb`
- Main pieces: `next_time`, `next_event`, `next`, `step!` and optional thinning callback

## 4) Generalized ensemble (multicanonical / Wang–Landau)


**Goal:** adapt weights online to flatten histogram / estimate DOS.

- Main pieces: `Multicanonical` or `WangLandau`, `update_weight!`, `update_f!`
- Start from equilibrium example and replace canonical weight with binned or mutable weight updates

## Suggested reading order

1. Framework
2. Core abstractions
3. Weights
4. Equilibrium or non-equilibrium algorithm page (depending on your project)
5. Measurements
6. This examples page + notebooks
