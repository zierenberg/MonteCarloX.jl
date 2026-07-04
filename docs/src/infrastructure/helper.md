# Helper Utilities
This page groups lower-level tools used by algorithms and model packages.

## Numeric helpers
- `log_sum`: numerically stable log-sum operations
- `binary_search`: utility for sorted-domain lookups
- `logistic`: logistic helper used in acceptance rules
- `kldivergence`: histogram/function divergence helper
```@docs
log_sum
binary_search
MonteCarloX.logistic
kldivergence
```

## RNG helper
`MutableRandomNumbers` is a lightweight utility for deterministic random-number
replay and control in workflows that need mutable RNG-like streams.
```@docs
MutableRandomNumbers
reset!(rng::MutableRandomNumbers, index::Int)
```

## Event-handler backends
These structures back event selection and queue-based scheduling in
continuous-time samplers.
```@docs
ListEventRateSimple
ListEventRateActiveMask
EventQueue
```