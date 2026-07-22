# ── General spin_flip! ───────────────────────────────────────────────────────
#
# One local update for any AbstractSpinSystem through the standard hooks (propose_state →
# delta_sys → delta_energy → accept! → modify!). The model hands the algorithm physical
# quantities, never a logweight: a linear ensemble (Boltzmann) takes the O(1) local ΔE
# (accept!(alg, ΔE)); a nonlinear ensemble (multicanonical, Wang-Landau) takes the absolute
# energies around the move (accept!(alg, E_new, E_old) — also drives visit recording). The
# algorithm owns the target π; energy stays a package-level concept.
#
# The single-site update is the primitive: spin_flip!(sys, alg, i) updates site i. The common
# Monte Carlo step spin_flip!(sys, alg) is the thin wrapper that draws a uniform site with
# pick_site. Making the site explicit is what enables deterministic schemes — a typewriter
# sweep is just `for i in eachindex(sys.spins); spin_flip!(sys, alg, i); end`.
#
# Discrete spins need no proposal parameter. A continuous spin takes a step width at the update
# call (kept there so it stays adaptable) through its OWN spin_flip! method — e.g. XY's rotation
# half-width: spin_flip!(sys, alg; Δθ) or spin_flip!(sys, alg, i; Δθ). The width is forwarded to
# that spin type's propose_state; a discrete propose_state simply takes no extra argument.

# logR assembly shared by spin_flip! and spin_exchange!. linear_logweight(ens) is a
# compile-time constant, so the branch folds away and the linear path stays allocation-free.
@inline function _accept_delta!(alg::MetropolisHastingsAlgorithm, sys, ΔE)
    if linear_logweight(ensemble(alg))
        return accept!(alg, ΔE)
    else
        E_old = energy(sys)
        return accept!(alg, E_old + ΔE, E_old)
    end
end

# Apply a proposed new state at `site` through the shared hooks (delta_sys → accept → modify!).
@inline function _flip!(sys, alg::MetropolisHastingsAlgorithm, site, s_new)
    δs = delta_sys(sys, site, s_new)
    ΔE = delta_energy(sys, site, s_new, δs)
    _accept_delta!(alg, sys, ΔE) && modify!(sys, site, s_new, δs)
    return nothing
end

# Parameter-free proposal (discrete spins): site primitive + random-site wrapper.
@inline spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm, site::Integer) =
    _flip!(sys, alg, site, propose_state(alg.rng, sys, site))
@inline spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm) =
    spin_flip!(sys, alg, pick_site(alg.rng, length(sys.spins)))

# XY: the rotation half-width Δθ lives at the update call — same site-primitive/random-wrapper
# split, with Δθ forwarded to the XY propose_state.
@inline spin_flip!(sys::SpinSystem{<:Any, <:XYSpin}, alg::MetropolisHastingsAlgorithm, site::Integer; Δθ::Real) =
    _flip!(sys, alg, site, propose_state(alg.rng, sys, site, Δθ))
@inline spin_flip!(sys::SpinSystem{<:Any, <:XYSpin}, alg::MetropolisHastingsAlgorithm; Δθ::Real) =
    spin_flip!(sys, alg, pick_site(alg.rng, length(sys.spins)); Δθ)

# ── Heat bath ─────────────────────────────────────────────────────────────────
#
# The conditional itself is the core Gibbs primitive `resample!` (generic over the site
# interface — no logweight in this package). Same site-primitive/random-wrapper split as
# above: the primitive decides, this update applies (raw-state modify!).

@inline function spin_flip!(sys::SpinSystem{<:Any, <:Spin}, alg::HeatBathAlgorithm, site::Integer)
    s_new = resample!(alg, sys, site)
    s_new === nothing || modify!(sys, site, s_new)
    return nothing
end

@inline spin_flip!(sys::SpinSystem{<:Any, <:Spin}, alg::HeatBathAlgorithm) =
    spin_flip!(sys, alg, pick_site(alg.rng, length(sys.spins)))
