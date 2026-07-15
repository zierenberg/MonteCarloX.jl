"""
    AbstractSpinSystem <: AbstractSystem

Base type for spin systems. The update functions drive any subtype through the hooks

    propose_state(rng, sys, i; proposal...) -> new spin state (keyword proposal parameters,
                                           e.g. the XY rotation half-width Δθ)
    delta_sys(sys, i, s_new)            -> delta payload (defaults to `nothing`)
    delta_energy(sys, i, s_new, δs)     -> energy change (default ignores the payload)
    modify!(sys, i, s_new, δs)          -> apply change (default ignores the payload)

The payload lets a system compute per-term deltas ONCE and reuse them in both the accept
decision and the commit. The four-argument `delta_energy`/`modify!` are called only by the
system-level updates; the three-argument `modify!(sys, i, s_new)` stays the raw-state entry
point (heat bath, n-fold way, tests).
"""
abstract type AbstractSpinSystem <: AbstractSystem end

"""
    delta_sys(sys, i, s_new)

Prepare a system-specific delta payload for setting site `i` to `s_new`. Default: `nothing`
(the fallback hooks below then route to the raw-state methods).
"""
@inline delta_sys(sys::AbstractSpinSystem, i, s_new) = nothing

@inline delta_energy(sys::AbstractSpinSystem, i, s_new, ::Nothing) = delta_energy(sys, i, s_new)
@inline MonteCarloX.modify!(sys::AbstractSpinSystem, i, s_new, ::Nothing) = modify!(sys, i, s_new)

"""
    pick_site(rng, N)

Randomly pick a site index from 1 to N.
"""
@inline pick_site(rng, N) = Int(rand(rng, UInt) % UInt(N)) + 1
