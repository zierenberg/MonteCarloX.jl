# ── General spin_flip! ───────────────────────────────────────────────────────
#
# One local update for any AbstractSpinSystem through the standard hooks (propose_state →
# delta_sys → delta_energy → accept! → modify!). The accept step assembles the core
# primitive's argument LOCALLY: a linear ensemble (Boltzmann) uses the O(1) local ΔE
# (accept!(alg, logR)); a nonlinear ensemble (multicanonical, Wang-Landau) reads the
# absolute energies around the move (accept!(alg, E_new, E_old) — also drives visit
# recording). No core wrapper: logR assembly is caller business, and energy stays a
# package-level concept.
#
# The single-site update is the primitive: spin_flip!(sys, alg, i) updates site i. The common
# Monte Carlo step spin_flip!(sys, alg) is the thin wrapper that draws a uniform site with
# pick_site. Making the site explicit is what enables deterministic schemes — a typewriter
# sweep is just `for i in eachindex(sys.spins); spin_flip!(sys, alg, i); end`.
#
# Proposal parameters ride as keywords, so the positional site stays unambiguous:
# spin_flip!(sys, alg; Δθ=0.3) (or spin_flip!(sys, alg, i; Δθ=0.3)) drives the XY rotation of
# half-width Δθ. They live with the update call, not the system, so they stay adaptable
# (step-size control) without touching the system.

# logR assembly shared by spin_flip! and spin_exchange!. linear_logweight(ens) is a
# compile-time constant, so the branch folds away and the linear path stays allocation-free.
@inline function _accept_delta!(alg::MetropolisHastingsAlgorithm, sys, ΔE)
    ens = ensemble(alg)
    if linear_logweight(ens)
        return accept!(alg, logweight(ens, ΔE))
    else
        E_old = energy(sys)
        return accept!(alg, E_old + ΔE, E_old)
    end
end

# Primitive: one local update at a GIVEN site.
@inline function spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm, site::Integer; proposal...)
    s_new = propose_state(alg.rng, sys, site; proposal...)
    δs = delta_sys(sys, site, s_new)
    ΔE = delta_energy(sys, site, s_new, δs)
    _accept_delta!(alg, sys, ΔE) && modify!(sys, site, s_new, δs)
    return nothing
end

# Wrapper: the common single-flip step at a uniformly drawn site.
@inline spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm; proposal...) =
    spin_flip!(sys, alg, pick_site(alg.rng, length(sys.spins)); proposal...)

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
