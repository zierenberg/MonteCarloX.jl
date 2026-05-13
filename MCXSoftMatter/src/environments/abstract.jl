"""
    AbstractEnvironment{D, T}

Geometry: metric + boundary. Three interface functions, one required.

| Function | Default | Override for |
|---|---|---|
| `difference(env, a, b)` | — (required) | all types (defines the metric) |
| `constrain(env, x)` | `x` (identity) | `PeriodicBox` (wrapping) |
| `is_valid(env, x)` | `true` | `HardWallBox`, `HardSphere`, complex geometry |
"""
abstract type AbstractEnvironment{D, T<:AbstractFloat} end

# ── Interface defaults ─────────────────────────────────────────────────────

@inline constrain(::AbstractEnvironment, x) = x
@inline is_valid(::AbstractEnvironment, x) = true

# ── Derived (never override) ───────────────────────────────────────────────

@inline distance_sq(env::AbstractEnvironment, a, b) = sum(abs2, difference(env, a, b))
@inline distance(env::AbstractEnvironment, a, b) = sqrt(distance_sq(env, a, b))
