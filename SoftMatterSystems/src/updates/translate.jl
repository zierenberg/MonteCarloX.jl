"""
    translate!(sys::ParticleGas, alg)

Single-particle displacement move. Picks a random particle, shifts it by a
random vector in [-delta, delta]^D, and accepts/rejects via the algorithm.
"""
function translate!(sys::ParticleGas{D,T}, alg::AbstractImportanceSampling) where {D,T}
    rng = alg.rng
    i = rand(rng, 1:sys.N)

    old_pos = sys.positions[i]
    E_old_i = _energy_of_particle(sys, i)

    displacement = SVector{D,T}(ntuple(_ -> sys.delta * (T(2) * rand(rng, T) - one(T)), Val(D)))
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
    translate!(sys::BeadSpringPolymer, alg; chain=false)

Displacement move for bead-spring polymers.
- `chain=false` (default): displace a single random monomer.
- `chain=true`: rigid translation of a random polymer chain.
"""
function translate!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg)
    else
        _translate_monomer!(sys, alg)
    end
end

function _translate_monomer!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling) where {D,T}
    rng = alg.rng
    n_total = sys.num_poly * sys.length_poly
    idx = rand(rng, 1:n_total)

    old_pos = sys.positions[idx]
    E_old = _monomer_energy(sys, idx)

    displacement = SVector{D,T}(ntuple(_ -> sys.delta * (T(2) * rand(rng, T) - one(T)), Val(D)))
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

function _translate_chain!(sys::BeadSpringPolymer{D,T}, alg::AbstractImportanceSampling) where {D,T}
    rng = alg.rng
    M = sys.length_poly
    n = rand(rng, 1:sys.num_poly)
    start_idx = (n - 1) * M + 1

    displacement = SVector{D,T}(ntuple(_ -> sys.delta * (T(2) * rand(rng, T) - one(T)), Val(D)))

    # Save old positions and compute old energy (pair only -- bonds/bending unchanged)
    old_positions = sys.positions[start_idx:start_idx + M - 1]
    E_old = zero(T)
    for k in 0:M-1
        E_old += _pair_energy_of_all(sys, start_idx + k)
    end

    # Apply displacement and update cell list
    for k in 0:M-1
        idx = start_idx + k
        new_pos = wrap_position(sys.positions[idx] + displacement, sys.L)
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
    end

    E_new = zero(T)
    for k in 0:M-1
        E_new += _pair_energy_of_all(sys, start_idx + k)
    end

    if accept!(alg, E_new, E_old)
        sys.cached_energy += E_new - E_old
    else
        for k in 0:M-1
            idx = start_idx + k
            sys.positions[idx] = old_positions[k + 1]
            update_particle!(sys.cell_list, idx, old_positions[k + 1])
        end
    end
    return nothing
end
