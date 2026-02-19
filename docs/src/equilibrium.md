## Importance Sampling

```@docs
Metropolis
Glauber
HeatBath
accept!
acceptance_rate
reset_statistics!
log_acceptance_ratio
```

## Generalized Ensembles

- `Multicanonical` uses `logWeight(E) = S(E) = log Ω(E)`.
- `WangLandau` uses density-of-states notation `g(E)` with
    `S(E) = log g(E)` and logarithmic schedule parameter `logf`.

```@docs
TabulatedLogWeight
Multicanonical
update_weights!
WangLandau
update_weight!
update_f!
```

## Reweighting

Reweighting tools are currently maintained outside the generated docs pages.

<!-- ## Full Docs
```@autodocs
Modules = [MonteCarloX]
Pages   = [
    "importance_sampling.jl",
    "generalized_ensemble.jl",
    "cluster_wolff.jl"
]
Private = false
``` -->