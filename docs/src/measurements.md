# Measurements
MonteCarloX separates **what** you measure from **when** you measure it.
`Measurement` and `Measurements` are convenience helpers — they are not required
for running MonteCarloX algorithms. You can always collect observables manually
in your simulation loop.

## Building blocks
- `Measurement`: one observable paired with one data container
- `Measurements`: named collection with a shared schedule
- schedule types:
  - `IntervalSchedule`
  - `PreallocatedSchedule`

## Step-based measurements (common in equilibrium loops)
```julia
using MonteCarloX

measurements = Measurements([
    :energy        => energy        => Float64[],
    :magnetization => magnetization => Float64[],
], interval=10)

# in your loop:
# measure!(measurements, sys, step)
```

This records every 10 loop steps.

## Time-based measurements (common in Gillespie loops)
```julia
using MonteCarloX

times = collect(0.0:0.1:10.0)
measurements = Measurements([
    :N => (state -> state[:N]) => Float64[],
], times)

# in your loop:
# measure!(measurements, state, t)
# stop criterion: is_complete(measurements)
```

`PreallocatedSchedule` handles event skipping by consuming all crossed checkpoints.

## Data access and reset
- `measurements[:energy].data` — raw storage
- `data(measurements, :energy)` — convenience accessor
- `reset!(measurements)` — clears data and restarts schedule state

## Integrated autocorrelation time
For replica-exchange or MCMC tuning, MonteCarloX provides a compact
integrated autocorrelation-time estimate:

```julia
using MonteCarloX

energies = data(measurements, :energy)
tauE = integrated_autocorrelation_time(energies)
# convenience alias:
# tauE = tau_int(energies)
```

This estimates

$$
tau_{\mathrm{int}} = \frac{1}{2} + \sum_{t\ge 1} C(t),
$$

with a self-consistent lag window. Use it as a practical tuning signal
for local decorrelation effort; short traces can be noisy. By default,
the maximal lag is capped at `floor(n/2)`.

## API reference
```@docs
Measurement
Measurements
MeasurementSchedule
IntervalSchedule
PreallocatedSchedule
measure!
reset!(measurement::Measurement)
reset!(schedule::IntervalSchedule)
reset!(measurements::Measurements)
integrated_autocorrelation_time
integrated_autocorrelation_times
tau_int
times(m::Measurements{K, PreallocatedSchedule}) where K
data(m::Measurements{K}, key::K) where K
is_complete
```