# ── General spin_flip! ───────────────────────────────────────────────────────
#
# Default implementation for any AbstractSpinSystem that provides:
#   propose_state(rng, sys, i) → new spin state
#   delta_sys(sys, i, s_new)   → local move delta payload
#   delta_energy(sys, i, dsys) → energy change
#   modify!(sys, i, dsys)      → apply change
#
# The accept step forms the log acceptance-ratio from the ensemble: a linear ensemble
# (Boltzmann) uses the O(1) local `ΔE` directly (`logR = logweight(ens, ΔE)`), while a nonlinear
# ensemble (multicanonical, Wang-Landau) reads the absolute energies around the move, because its
# logweight is not linear in `ΔE`. `linear_logweight(ens)` is a compile-time constant, so the
# branch folds away and the Boltzmann fast path stays allocation-free.
#
# Models may override spin_flip! for efficiency when needed.

# logR assembly shared by the general and Ising fast paths.
@inline function _accept_flip!(alg::MetropolisHastingsAlgorithm, sys, ΔE)
    ens = ensemble(alg)
    if linear_logweight(ens)
        return accept!(alg, logweight(ens, ΔE))
    else
        E_old = energy(sys)
        return accept!(alg, E_old + ΔE, E_old)
    end
end

@inline function spin_flip!(sys::AbstractSpinSystem, alg::MetropolisHastingsAlgorithm)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    dsys = delta_sys(sys, i, s_new)
    ΔE = delta_energy(sys, i, dsys)
    _accept_flip!(alg, sys, ΔE) && modify!(sys, i, dsys)
    return nothing
end

# ── Ising: specialized path skipping propose_state ──────────────────────────

@inline function spin_flip!(sys::AbstractIsing, alg::MetropolisHastingsAlgorithm)
    i = pick_site(alg.rng, length(sys.spins))
    dsys = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, dsys)
    _accept_flip!(alg, sys, ΔE) && modify!(sys, i, dsys)
    return nothing
end

# ── Heat bath: draw directly from the local conditional ∝ exp(logweight(ens, E(s'))) ──
#
# HeatBath now carries an ensemble, so the per-state weights go through `logweight` — one β-free
# expression that reduces to `exp(-β E)` for a BoltzmannEnsemble. For two states the conditional
# is the logistic (≡ Glauber balance).

@inline function spin_flip!(sys::AbstractIsing, alg::HeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    s_old = sys.spins[i]
    lpi = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, lpi)
    ens = ensemble(alg)
    # p(+1) = logistic(logweight(ens, E₋₁ − E₊₁)) with E₋₁ − E₊₁ = −s_old·ΔE
    p_plus = logistic(logweight(ens, -float(s_old) * ΔE))
    s_new = rand(alg.rng) < p_plus ? Int8(1) : Int8(-1)
    s_new != s_old && modify!(sys, i, lpi)
    alg.steps += 1
    return nothing
end

# ── BlumeCapel: HeatBath (enumerates 3 states) ──────────────────────────────

@inline function spin_flip!(sys::AbstractBlumeCapel, alg::HeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    coupling = neighbor_sum(sys, i)
    h_i = Float64(_site_field(sys.h, i))
    ens = ensemble(alg)

    e1 = coupling + h_i + float(sys.Δ)
    e2 = 0.0
    e3 = -coupling - h_i + float(sys.Δ)

    w1 = exp(logweight(ens, e1))
    w2 = exp(logweight(ens, e2))
    w3 = exp(logweight(ens, e3))
    z = w1 + w2 + w3

    r = rand(alg.rng) * z
    s_new = r < w1 ? Int8(-1) : (r < (w1 + w2) ? Int8(0) : Int8(1))
    s_new != sys.spins[i] && modify!(sys, i, delta_sys(sys, i, s_new))
    alg.steps += 1
    return nothing
end

# The composed nonreciprocal `SpinSystem` needs no bespoke method: with the temperature-independent
# vision-cone coupling J + κ its `delta_energy` is β-free, so it drives the generic
# `spin_flip!(::AbstractSpinSystem, ::MetropolisHastingsAlgorithm)` above via `propose_state`/`delta_energy`
# (defined in systems/nonreciprocal.jl), with β supplied by `logweight(ens, ΔE)` in the accept step.
