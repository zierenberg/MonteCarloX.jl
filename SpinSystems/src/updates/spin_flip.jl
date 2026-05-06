# ── General spin_flip! ───────────────────────────────────────────────────────
#
# Default implementation for any AbstractSpinSystem that provides:
#   propose_state(rng, sys, i) → new spin state
#   delta_energy(sys, i, s_new) → energy change
#   modify!(sys, i, s_new)     → apply change
#
# Models may override for efficiency (e.g., Ising skips propose_state).

function spin_flip!(sys::AbstractSpinSystem, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    ΔE = delta_energy(sys, i, s_new)
    accept!(alg, ΔE) && modify!(sys, i, s_new)
    return nothing
end

function spin_flip!(sys::AbstractSpinSystem, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    ΔE = delta_energy(sys, i, s_new)
    E_old = energy(sys)
    accept!(alg, E_old + ΔE, E_old) && modify!(sys, i, s_new)
    return nothing
end

# ── Ising: efficient dispatch (flip is implicit, single neighbor pass) ──────
#
# Compute local_pair_interactions once and reuse for both ΔE and modify!.

function spin_flip!(sys::AbstractIsing, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    lpi = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, lpi)
    accept!(alg, ΔE) && modify!(sys, i, lpi)
    return nothing
end

function spin_flip!(sys::AbstractIsing, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    lpi = local_pair_interactions(sys, i)
    ΔE = delta_energy(sys, i, lpi)
    E_old = energy(sys)
    accept!(alg, E_old + ΔE, E_old) && modify!(sys, i, lpi)
    return nothing
end

function spin_flip!(sys::AbstractIsing, alg::AbstractHeatBath)
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

function spin_flip!(sys::AbstractBlumeCapel, alg::AbstractHeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    coupling = _local_coupling(sys, i)
    h_i = Float64(_site_field(sys.h, i))

    e1 = coupling + h_i + float(sys.crystal)
    e2 = 0.0
    e3 = -coupling - h_i + float(sys.crystal)

    w1 = exp(-alg.β * e1)
    w2 = exp(-alg.β * e2)
    w3 = exp(-alg.β * e3)
    z = w1 + w2 + w3

    r = rand(alg.rng) * z
    s_new = r < w1 ? Int8(-1) : (r < (w1 + w2) ? Int8(0) : Int8(1))
    s_new != sys.spins[i] && modify!(sys, i, s_new)
    alg.steps += 1
    return nothing
end
