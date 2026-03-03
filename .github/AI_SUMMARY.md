# MonteCarloX AI Summary (Read First)

Core concept: **MonteCarloX must stay concise, modular, and compact**.

- Concise: minimal public API surface, avoid feature duplication.
- Modular: clear separation between algorithms, weights, measurements, and systems.
- Compact: prefer one general primitive over multiple special-case helpers.

MonteCarloX is algorithm-centric: model/system definitions are external to core.

## Architecture (current)

- Core abstractions: `AbstractSystem`, `AbstractAlgorithm`, `AbstractLogWeight`, `AbstractUpdate`, `AbstractMeasurement`.
- Equilibrium importance-sampling algorithms: `Metropolis`, `HeatBath`/`Glauber`, `Multicanonical`, `ParallelMulticanonical`, `WangLandau`.
- Non-equilibrium algorithms: `Gillespie` and kinetic Monte Carlo utilities.
- Weight representations: `BoltzmannLogWeight`, `BinnedLogWeight`.
- Measurement layer: `Measurement`, `Measurements`, schedules (`IntervalSchedule`, `PreallocatedSchedule`) and `measure!` / `reset!`.
- Event-handler infrastructure for rate/time-based event queues.

## Key API patterns

- Acceptance logic is centralized through `accept!` and shared counters (`steps`, `accepted`, `acceptance_rate`).
- Histogram/tabulated generalized-ensemble methods use `BinnedLogWeight`.
- Multicanonical uses one compact general setter:
	- `set_logweight!(alg, range, f)` where `f` is evaluated on bin centers in the selected range.
	- Use this primitive for analytic reference initialization and boundary-tail shaping.

## Testing status and scope

- Tests are organized by module under `test/` and run from `test/runtests.jl`.
- Current multicanonical-focused tests pass, including the range/function `set_logweight!` behavior.

## Implementation principle

- New API design should default to the smallest general interface.
- If a proposal can be expressed by composing existing primitives, do not add new top-level API.
- Keep framework code free of model-specific assumptions.

## Guidance for contributors and agents

- Do not add model-specific logic to core framework code.
- When adding features, first look for a smaller general API instead of multiple task-specific entry points.
- Keep examples/notebooks aligned with exported APIs to avoid duplicated ad-hoc logic.
