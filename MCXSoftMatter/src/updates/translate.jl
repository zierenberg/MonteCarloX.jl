"""
    translate!(sys::ParticleGas, alg, delta)

Single-particle displacement move. Picks a random particle, shifts it by a
random vector in [-Δ, Δ]^D, and accepts/rejects via the algorithm.
"""
function translate!(sys::ParticleGas{D,T}, alg::AbstractImportanceSampling, Δ::T) where {D,T}
    rng = alg.rng
    i = rand(rng, 1:sys.N)

    old_pos = sys.positions[i]
    E_old_i = _energy_of_particle(sys, i)

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))
    new_pos = wrap_position(old_pos + displacement, sys.L)
    sys.positions[i] = new_pos
    update_particle!(sys.cell_list, i, new_pos)

    E_new_i = _energy_of_particle(sys, i)

    if accept!(alg, E_new_i, E_old_i)
        sys.cached_energy += E_new_i - E_old_i
    else
        sys.positions[i] = old_pos
        update_particle!(sys.cell_list, i, old_pos)
    end
    return nothing
end

"""
    translate!(sys::BeadSpringPolymer, alg, Δ; chain=false)

Displacement move for bead-spring polymers.
- `chain=false` (default): displace a single random monomer.
- `chain=true`: rigid translation of a random polymer chain.
"""
function translate!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling, Δ::T; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_monomer!(sys, alg, Δ)
    end
end

function _translate_monomer!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling, Δ::T) where {D,T}
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
        sys.cached_energy += E_new - E_old
    else
        sys.positions[idx] = old_pos
        update_particle!(sys.cell_list, idx, old_pos)
    end
    return nothing
end

function _translate_chain!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling, Δ::T) where {D,T}
    rng = alg.rng
    n = rand(rng, 1:sys.num_poly)
    M         = sys.lengths[n]
    start_idx = sys.offsets[n] + 1

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))

    # Pair-only energy is sufficient for rigid-chain translation.
    E_old = zero(T)
    @inbounds for k in 1:M
        E_old += _pair_energy_of_all(sys, start_idx + k - 1)
    end

    # Apply displacement and update cell list
    @inbounds for k in 1:M
        idx     = start_idx + k - 1
        new_pos = wrap_position(sys.positions[idx] + displacement, sys.L)
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
    end

    E_new = zero(T)
    @inbounds for k in 1:M
        E_new += _pair_energy_of_all(sys, start_idx + k - 1)
    end

    if accept!(alg, E_new, E_old)
        sys.cached_energy += E_new - E_old
    else
        # Invert the displacement: wrap_position(new_pos - displacement, L) == old_pos
        @inbounds for k in 1:M
            idx     = start_idx + k - 1
            old_pos = wrap_position(sys.positions[idx] - displacement, sys.L)
            sys.positions[idx] = old_pos
            update_particle!(sys.cell_list, idx, old_pos)
        end
    end
    return nothing
end
