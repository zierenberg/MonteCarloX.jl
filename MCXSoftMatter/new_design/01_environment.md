# Environment

Geometry: metric + boundary. Three functions, one required.

```julia
abstract type AbstractEnvironment{D,T} end
```

## Interface

| Function | Default | Override for |
|---|---|---|
| `difference(env, a, b)` | — (required) | all types (defines the metric) |
| `constrain(env, x)` | `x` (identity) | `PeriodicBox` (wrapping) |
| `is_valid(env, x)` | `true` | `HardWallBox`, `HardSphere`, complex geometry |

Derived (never override):
```julia
@inline distance(env, a, b) = sqrt(sum(abs2, difference(env, a, b)))
```

### Usage in updates

```julia
x_new = constrain(env, x_old + Δ)
is_valid(env, x_new) || return nothing
ΔE = local_energy(sys, idx, x_new) - local_energy(sys, idx, x_old)
accept!(alg, ΔE) && (positions[idx] = x_new; ...)
```

- `constrain` always returns a position (wrapping for PBC, identity otherwise).
- `is_valid` always returns a bool (`true` by default, hard geometry overrides).
- For `PeriodicBox`/`FreeSpace`: `is_valid` returns `true` unconditionally → compiled away.
- Updates are **environment-agnostic** — dispatch happens inside these functions.

Soft confinement (harmonic traps etc.) is an energy contribution on the system,
not part of the environment. See `02_system_model.md`.

---

## Concrete types

```julia
struct PeriodicBox{D, T<:AbstractFloat} <: AbstractEnvironment{D,T}
    L::T
end
@inline difference(env::PeriodicBox, a, b) = a - b - env.L * round((a - b) / env.L)
@inline constrain(env::PeriodicBox, x)     = x - env.L * floor.(x / env.L)

struct FreeSpace{D, T<:AbstractFloat} <: AbstractEnvironment{D,T} end
@inline difference(::FreeSpace, a, b) = a - b

struct HardWallBox{D, T<:AbstractFloat} <: AbstractEnvironment{D,T}
    L::T
end
@inline difference(::HardWallBox, a, b) = a - b
@inline is_valid(env::HardWallBox{D}, x) where D =
    all(d -> 0 <= x[d] < env.L, 1:D)
```

---

## Future

- **Non-cubic box**: generalize `L::T` to `L::SVector{D,T}`.
- **`HardSphere`**: `is_valid` checks distance from center < R.
- **Complex geometry** (channels, cavities): custom `is_valid` on a user-defined environment type.
- **Slit geometry**: periodic in x/y, confined in z → composite type or dedicated implementation.
