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
# Extra arguments are forwarded to the proposal: spin_flip!(sys, alg, Δθ) drives the XY
# rotation of half-width Δθ — proposal parameters live with the update call, not with the
# system, so they stay adaptable (step-size control) without touching the system.

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

@inline function spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm, proposal_args...)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i, proposal_args...)
    δs = delta_sys(sys, i, s_new)
    ΔE = delta_energy(sys, i, s_new, δs)
    _accept_delta!(alg, sys, ΔE) && modify!(sys, i, s_new, δs)
    return nothing
end

# ── Heat bath ─────────────────────────────────────────────────────────────────
#
# The conditional itself is the core Gibbs primitive `resample!` (generic over the site
# interface — no logweight in this package). Same template as the accept branch: the
# primitive decides, this update applies (raw-state modify!).

@inline function spin_flip!(sys::SpinSystem{<:Any, <:Spin}, alg::HeatBathAlgorithm)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = resample!(alg, sys, i)
    s_new === nothing || modify!(sys, i, s_new)
    return nothing
end
