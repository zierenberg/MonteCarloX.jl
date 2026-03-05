# Continuous-Time Sampling Algorithms

Continuous-time Monte Carlo in MonteCarloX is event-driven.
You sample **when** the next event happens and **which** event happens.

## Why this differs from Metropolis-style sampling

- Importance sampling focuses on stationary distributions.
- Continuous-time sampling focuses on trajectories in real/simulation time.

## Gillespie workflow

At each step:

1. compute total rate
2. sample waiting time `dt ~ Exp(total_rate)`
3. sample event index proportional to rates
4. advance time and apply event update

`Gillespie` stores `steps` and current `time`.

## Minimal loop

```julia
using Random
using MonteCarloX

alg = Gillespie(MersenneTwister(10))
rates = [0.5, 1.0]  # e.g. birth, death

for _ in 1:10000
	t, event = step!(alg, rates)
	# apply event to your state
	# update rates from new state
end
```

## `step!` vs `advance!`

- `step!`: one event at a time (full manual control)
- `advance!`: run until `total_time`, with optional callbacks (`measure!`, `update!`)

Example with explicit state + rates callback:

```julia
using Random
using MonteCarloX

sys = Dict(:N => 30)
rates(sys, t) = [0.2 * sys[:N], 0.1 * sys[:N]]

alg = Gillespie(MersenneTwister(11))

update_cb = (state, event, t) -> begin
	if event == 1
		state[:N] += 1
	elseif event == 2
		state[:N] = max(0, state[:N] - 1)
	end
end

advance!(alg, sys, 20.0; rates=rates, update!=update_cb)
```

## Event sources supported

- raw vectors of rates
- `StatsBase` weights
- rate-based event handlers (`AbstractEventHandlerRate`)
- time-ordered event queues (`AbstractEventHandlerTime`)
- callback `rates_at_time(t)`

## Low-level building blocks

- `next_time`: waiting-time sampling (homogeneous or thinning-based)
- `next_event`: event-index sampling
- `next`: one combined draw `(dt, event)`

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