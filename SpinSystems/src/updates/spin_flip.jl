# ── General spin_flip! ───────────────────────────────────────────────────────
#
# Default implementation for any AbstractSpinSystem that provides:
#   propose_state(rng, sys, i) → new spin state
#   delta_sys(sys, i, s_new)   → local move delta payload
#   delta_energy(sys, i, dsys) → energy change
#   modify!(sys, i, dsys)      → apply change
#
# Models may override for efficiency when needed.

@inline function spin_flip!(sys::AbstractSpinSystem, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    dsys = delta_sys(sys, i, s_new)
    ΔE = delta_energy(sys, i, dsys)
    accept!(alg, ΔE) && modify!(sys, i, dsys)
    return nothing
end

@inline function spin_flip!(sys::AbstractSpinSystem, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    dsys = delta_sys(sys, i, s_new)
    ΔE = delta_energy(sys, i, dsys)
    E_old = energy(sys)
    accept!(alg, E_old + ΔE, E_old) && modify!(sys, i, dsys)
    return nothing
end

# ── Ising: specialized path skipping propose_state ──────────────────────────

@inline function spin_flip!(sys::AbstractIsing, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    dsys = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, dsys)
    accept!(alg, ΔE) && modify!(sys, i, dsys)
    return nothing
end

@inline function spin_flip!(sys::AbstractIsing, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    dsys = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, dsys)
    E_old = energy(sys)
    accept!(alg, E_old + ΔE, E_old) && modify!(sys, i, dsys)
    return nothing
end

@inline function spin_flip!(sys::AbstractIsing, alg::AbstractHeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    s_old = sys.spins[i]
    lpi = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, lpi)
    p_plus = logistic(alg.β * float(s_old) * ΔE)
    s_new = rand(alg.rng) < p_plus ? Int8(1) : Int8(-1)
    s_new != s_old && modify!(sys, i, lpi)
    alg.steps += 1
    return nothing
end

# ── BlumeCapel: HeatBath (enumerates 3 states) ──────────────────────────────

@inline function spin_flip!(sys::AbstractBlumeCapel, alg::AbstractHeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    coupling = _local_coupling(sys, i)
    h_i = Float64(_site_field(sys.h, i))

    e1 = coupling + h_i + float(sys.Δ)
    e2 = 0.0
    e3 = -coupling - h_i + float(sys.Δ)

    w1 = exp(-alg.β * e1)
    w2 = exp(-alg.β * e2)
    w3 = exp(-alg.β * e3)
    z = w1 + w2 + w3

    r = rand(alg.rng) * z
    s_new = r < w1 ? Int8(-1) : (r < (w1 + w2) ? Int8(0) : Int8(1))
    s_new != sys.spins[i] && modify!(sys, i, delta_sys(sys, i, s_new))
    alg.steps += 1
    return nothing
end
