function translate!(sys::ParticleSystem{D,T}, alg::AbstractImportanceSampling, Δ::T; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_monomer!(sys, alg, Δ)
    end
end

function _translate_monomer!(sys::ParticleSystem{D,T}, alg::AbstractImportanceSampling, Δ::T) where {D,T}
    rng = alg.rng
    idx = rand(rng, 1:length(sys.positions))

    old_pos = sys.positions[idx]
    E_old = _monomer_energy(sys, idx)

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))
    new_pos = wrap_position(old_pos + displacement, sys.L)
    sys.positions[idx] = new_pos
    update_particle!(sys.cell_list, idx, new_pos)

    E_new = _monomer_energy(sys, idx)

    if accept!(alg, E_new, E_old)
        sys.cache.pair += E_new - E_old
    else
        sys.positions[idx] = old_pos
        update_particle!(sys.cell_list, idx, old_pos)
    end
    return nothing
end

function _translate_chain!(sys::ParticleSystem{D,T,P,<:Polymer}, alg::AbstractImportanceSampling, Δ::T) where {D,T,P}
    rng = alg.rng
    n = rand(rng, 1:length(sys.molecules))
    mol = sys.molecules[n]
    M         = mol.length
    start_idx = mol.offset + 1

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))

    E_old = zero(T)
    @inbounds for k in 1:M
        E_old += _local_pair_energy_no_excl(sys, start_idx + k - 1)
    end

    @inbounds for k in 1:M
        idx     = start_idx + k - 1
        new_pos = wrap_position(sys.positions[idx] + displacement, sys.L)
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
    end

    E_new = zero(T)
    @inbounds for k in 1:M
        E_new += _local_pair_energy_no_excl(sys, start_idx + k - 1)
    end

    if accept!(alg, E_new, E_old)
        sys.cache.pair += E_new - E_old
    else
        @inbounds for k in 1:M
            idx     = start_idx + k - 1
            old_pos = wrap_position(sys.positions[idx] - displacement, sys.L)
            sys.positions[idx] = old_pos
            update_particle!(sys.cell_list, idx, old_pos)
        end
    end
    return nothing
end
