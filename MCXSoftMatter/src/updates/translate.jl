function translate!(sys::ParticleSystem{D,T}, alg::AbstractImportanceSampling, Δ::T; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_monomer!(sys, alg, Δ)
    end
end

function translate!(sys::ParticleSystem{D,T}, alg::AbstractMetropolis, Δ::T; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_monomer!(sys, alg, Δ)
    end
end

# ── Monomer translate ─────────────────────────────────────────────────────────

function _translate_monomer!(sys::ParticleSystem{D,T}, alg::AbstractMetropolis, Δ::T) where {D,T}
    rng = alg.rng
    idx = rand(rng, 1:length(sys.positions))

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))
    new_pos = constrain(sys.env, sys.positions[idx] + displacement)

    dE = _monomer_energy_change(sys, idx, new_pos)
    if accept!(alg, _total_dE(dE))
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
        _update_cache!(sys.cache, dE)
    end
    return nothing
end

function _translate_monomer!(sys::ParticleSystem{D,T}, alg::AbstractImportanceSampling, Δ::T) where {D,T}
    rng = alg.rng
    idx = rand(rng, 1:length(sys.positions))

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))
    new_pos = constrain(sys.env, sys.positions[idx] + displacement)

    dE = _monomer_energy_change(sys, idx, new_pos)
    E_old = energy(sys)
    E_new = E_old + _total_dE(dE)

    if accept!(alg, E_new, E_old)
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
        _update_cache!(sys.cache, dE)
    end
    return nothing
end

# ── Chain translate ───────────────────────────────────────────────────────────

function _translate_chain!(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, alg::AbstractMetropolis, Δ::T) where {D,T,TEnv,P}
    rng = alg.rng
    n = rand(rng, 1:length(sys.molecules))
    mol = sys.molecules[n]
    M         = mol.length
    start_idx = mol.offset + 1

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))

    dE = _pair_energy_change(sys, start_idx, M, displacement)
    if accept!(alg, dE)
        @inbounds for k in 1:M
            idx     = start_idx + k - 1
            new_pos = constrain(sys.env, sys.positions[idx] + displacement)
            sys.positions[idx] = new_pos
            update_particle!(sys.cell_list, idx, new_pos)
        end
        sys.cache.pair += dE
    end
    return nothing
end

function _translate_chain!(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, alg::AbstractImportanceSampling, Δ::T) where {D,T,TEnv,P}
    rng = alg.rng
    n = rand(rng, 1:length(sys.molecules))
    mol = sys.molecules[n]
    M         = mol.length
    start_idx = mol.offset + 1

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))

    dE = _pair_energy_change(sys, start_idx, M, displacement)
    E_old = energy(sys)
    E_new = E_old + dE

    if accept!(alg, E_new, E_old)
        @inbounds for k in 1:M
            idx     = start_idx + k - 1
            new_pos = constrain(sys.env, sys.positions[idx] + displacement)
            sys.positions[idx] = new_pos
            update_particle!(sys.cell_list, idx, new_pos)
        end
        sys.cache.pair += dE
    end
    return nothing
end
