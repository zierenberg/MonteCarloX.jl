## Kinetic Monte Carlo (continuous-time)

`Gillespie` is the simplest user-facing continuous-time algorithm in MonteCarloX.
The general stepping API is:

1. `dt, event = next(alg, source)`
2. advance time by `dt`
3. (optional) measure at the new time
4. apply state update for `event`

This makes it straightforward to place measurements before state modification,
which is often required in kinetic simulations.

### Sources for `next`

- raw rates (`AbstractVector`)
- weighted rates (`AbstractWeights`)
- event handlers (`AbstractEventHandlerRate`)
- time-dependent rate callback with `next(alg, rates_at_time)` where `rates_at_time(alg.time)` returns rates

### `advance!` callbacks

For model systems, `advance!` supports both callbacks:

- `measure!(sys, t_new, event)` called first
- `update!(sys, event, t_new)` called second

and requires an explicit rate callback keyword:

- `rates = (sys, t) -> ...`

For rates/event-handler sources, `advance!` supports:

- `update!(source, event, t_new)`

### Poisson via kinetic Monte Carlo

Poisson processes are implemented directly with the kinetic Monte Carlo
primitives (`next_time`, `next_event`, `next`). See the notebook
`notebooks/poisson_kmc.ipynb` for a complete homogeneous and inhomogeneous
example using thinning.

```@docs
Gillespie
next
next_time
next_event
advance!
```

<!-- ## Full Docs

```@autodocs
Modules = [MonteCarloX]
Pages   = [
    "gillespie.jl",
]
Private = false
``` -->