# Core Abstractions

This page explains the abstractions as responsibilities, not just type names.

## `AbstractSystem`

What it owns:

- state variables
- model parameters
- model-specific observables (for example energy)
- model-specific local update ingredients

What it should not own:

- generic sampling logic

## `AbstractLogWeight`

Encodes relative probability for equilibrium sampling.

- Canonical ensemble: `BoltzmannLogWeight(β)`
- Generalized ensemble: tabulated `BinnedLogWeight`

Algorithms only need local log-weight differences, so they remain model-agnostic.

## `AbstractAlgorithm`

Owns sampler state and statistics.

Examples:

- `Metropolis`, `Glauber`, `HeatBath`
- `Multicanonical`, `WangLandau`
- `Gillespie`

Typical fields are RNG, counters (`steps`, `accepted`) and/or simulation time.

## `AbstractUpdate`

Represents update mechanics.
In practice, this usually appears as system-side methods like `spin_flip!` that call algorithm primitives (`accept!`, `step!`, ...).

## `AbstractMeasurement`

Represents observable extraction and storage.
MonteCarloX provides reusable scheduling through `Measurements`.

## Practical checklist for a new model package

To integrate with MonteCarloX cleanly, provide:

1. A concrete `AbstractSystem`
2. Observable functions you care about
3. Update function(s) that call MonteCarloX algorithm primitives
4. Initialization helpers (`init!`-style)

## API reference

```@docs
AbstractSystem
AbstractLogWeight
AbstractAlgorithm
AbstractUpdate
AbstractMeasurement
```
