# Measurements

Measurements are first-class objects in MonteCarloX.
You define observables once and choose a schedule separately.

## Building a measurement set

```julia
using MonteCarloX

measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)
```

This uses `IntervalSchedule` and records every 10 steps (or time units,
depending on your loop variable).

## Preallocated schedule

For event-driven simulations, pre-defined measurement times are often better:

```julia
times = collect(0.0:0.1:10.0)
measurements = Measurements([
    :x => state_value => Float64[]
], times)

# in simulation loop
measure!(measurements, system, t)
if is_complete(measurements)
    # all target times were measured
end
```

## API reference

```@docs
Measurement
Measurements
MeasurementSchedule
IntervalSchedule
PreallocatedSchedule
measure!
is_complete
```
