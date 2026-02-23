# Continuous-Time Sampling Algorithms

Non-equilibrium simulations are event-driven in MonteCarloX.
The core primitive is kinetic Monte Carlo in continuous time.

This is distinct from importance sampling:

- **Importance sampling** targets a stationary distribution via accept/reject updates.
- **Continuous-time sampling** advances stochastic dynamics with sampled waiting times and events.

## Continuous-time stepping model

The standard loop is:

1. start from current state,
2. compute/provide event rates,
3. sample `(dt, event)` (which advances simulation time),
4. measure at the new time,
5. apply event update to state,
6. refresh rates/event structure if needed.

`Gillespie` is the canonical user-facing algorithm for this pattern.

## Sources for event sampling

`next` / `step!` work with multiple backends:

- raw rates (`AbstractVector`)
- weighted rates (`AbstractWeights`)
- event handlers (`AbstractEventHandlerRate`, `AbstractEventHandlerTime`)
- time-dependent rate callback (`rates_at_time(alg.time) -> rates`)

## Minimal example

```julia
using Random
using MonteCarloX

rng = MersenneTwister(11)
alg = Gillespie(rng)

rates = [0.1, 0.2, 0.05]

for _ in 1:10_000
	t, event = step!(alg, rates)
	# measure at time t
	# apply event to your model state
	# update rates if state changed
end
```

## Poisson process perspective

Homogeneous and inhomogeneous Poisson processes can be implemented directly with
`next_time`, `next_event`, `next`, and optional thinning callbacks.
See `notebooks/poisson_kmc.ipynb` for a full worked notebook.

## API reference

```@docs
AbstractKineticMonteCarlo
Gillespie
next
step!
next_time
next_event
advance!
reset!(alg::AbstractKineticMonteCarlo)
```