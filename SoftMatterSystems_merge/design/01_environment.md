# Environment

Metric + boundary enforcement. Confinement potentials are interactions, not geometry.

```julia
abstract type AbstractEnvironment{D,T} end
```

Required interface — four functions:

| Function | Returns | Meaning |
|---|---|---|
| `difference(env, a, b)` | `SVector{D,T}` | shortest vector **from b to a** (minimum-image if periodic) |
| `distance(env, a, b)` | `T` | scalar distance; derived: `sqrt(sum(abs2, difference(env,a,b)))` |
| `move(env, x, Δ)` | `SVector{D,T}` | apply displacement Δ to position x and enforce boundary: `wrap(env, x + Δ)` |
| `is_valid(env, x)` | `Bool` | only needed for hard walls; default `true` |

`difference` and `distance` are the metric. `move` is how particles are displaced —
it wraps the result into the domain so callers never call `wrap` manually.

```julia
# Default implementations (override only if needed)
@inline distance(env, a, b)  = sqrt(sum(abs2, difference(env, a, b)))
@inline move(env, x, Δ)      = wrap(env, x + Δ)
```

A trial move in any update function reduces to:
```julia
x_new = move(env, x_old, Δ)   # propose
is_valid(env, x_new) || return  # hard-wall rejection (no-op for PeriodicBox)
ΔE = local_energy(sys, idx, x_new) - local_energy(sys, idx, x_old)
accept!(alg, ΔE) ? (positions[idx] = x_new) : nothing
```

---

## Concrete types

```julia
# Primary. Cell list lives inside — chosen automatically from cutoff and N.
struct PeriodicBox{D, T<:AbstractFloat, Nbr} <: AbstractEnvironment{D,T}
    L::T
    neighbors::Nbr   # CellList{D,K} | NoCellList
end

function PeriodicBox{D,T}(L; pair_potential=nothing, N=nothing) where {D,T}
    nbr = _build_neighbor_backend(Val(D), T(L), N, pair_potential)
    PeriodicBox{D,T,typeof(nbr)}(T(L), nbr)
end

@inline difference(env::PeriodicBox{D,T}, a, b) where {D,T} =
    a - b - env.L * round((a - b) / env.L)
@inline wrap(env::PeriodicBox{D,T}, x) where {D,T} =
    x - env.L * floor.(x / env.L)

# Open boundary — single-chain studies, cluster sampling.
struct FreeSpace{D, T<:AbstractFloat} <: AbstractEnvironment{D,T} end

@inline difference(::FreeSpace, a, b) = a - b
@inline wrap(::FreeSpace, x)          = x
```

**Future**: `HardWallBox` — just `is_valid(env, x) = all(0 .<= x .< env.L)`; MC rejects, no reflection.
Non-cubic box: generalise `L::T` to `L::SVector{D,T}`; only `displacement`/`wrap` change.
