# Core Abstractions

These abstractions are the stable conceptual backbone of MonteCarloX.

## `AbstractSystem`

Represents the model state (configuration, constraints, cached observables, etc.).
The system decides **what** can happen and how to compute quantities such as energy.

## `AbstractLogWeight`

Represents the target log-weight for equilibrium sampling.
For canonical simulations this is usually Boltzmann-like.
For generalized ensembles this can be tabulated and updated online.

## `AbstractAlgorithm`

Represents the Monte Carlo engine (Metropolis, HeatBath, Gillespie, ...).
It usually stores RNG and statistics (`steps`, acceptance counters, simulation time).

## `AbstractUpdate`

Represents update logic coordinating algorithm and state evolution.
In practice, user-facing updates are often model functions (e.g. `spin_flip!`).

## `AbstractMeasurement`

Represents observable extraction and storage.
MonteCarloX provides `Measurement` and `Measurements` with scheduling.

## API reference

```@docs
AbstractSystem
AbstractLogWeight
AbstractAlgorithm
AbstractUpdate
AbstractMeasurement
```
