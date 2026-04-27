"""
    monomer_move!(sys::BeadSpringPolymer, alg::AbstractMetropolis)

Single-monomer displacement move for a bead-spring polymer system.
Picks a random monomer, shifts it by a random vector in [-delta, delta]³,
and accepts/rejects via Metropolis criterion.

Energy change is computed by evaluating only the terms involving the
displaced monomer (pair, bond, bending contributions).
"""
function monomer_move!(sys::BeadSpringPolymer{T}, alg::AbstractMetropolis) where T
    rng = alg.rng
    n_total = sys.M * sys.N
    idx = rand(rng, 1:n_total)

    old_pos = sys.positions[idx]

    # Compute old energy contribution of this monomer
    E_old = _monomer_energy(sys, idx, sys.positions[idx])

    # Propose displacement
    dx = sys.delta * (T(2) * rand(rng, T) - one(T))
    dy = sys.delta * (T(2) * rand(rng, T) - one(T))
    dz = sys.delta * (T(2) * rand(rng, T) - one(T))
    new_pos = wrap_position(old_pos + SVector{3,T}(dx, dy, dz), sys.L)
    sys.positions[idx] = new_pos

    # Compute new energy contribution
    E_new = _monomer_energy(sys, idx, new_pos)
    ΔE = E_new - E_old

    if accept!(alg, ΔE)
        # Update cached energies via full recomputation
        _recompute_energy!(sys)
    else
        sys.positions[idx] = old_pos
    end
    return nothing
end

"""
    monomer_move!(sys::BeadSpringPolymer, alg::AbstractImportanceSampling)

Single-monomer displacement with generic importance sampling.
"""
function monomer_move!(sys::BeadSpringPolymer{T}, alg::AbstractImportanceSampling) where T
    rng = alg.rng
    n_total = sys.M * sys.N
    idx = rand(rng, 1:n_total)

    E_old_total = energy(sys)
    old_pos = sys.positions[idx]

    dx = sys.delta * (T(2) * rand(rng, T) - one(T))
    dy = sys.delta * (T(2) * rand(rng, T) - one(T))
    dz = sys.delta * (T(2) * rand(rng, T) - one(T))
    new_pos = wrap_position(old_pos + SVector{3,T}(dx, dy, dz), sys.L)
    sys.positions[idx] = new_pos

    _recompute_energy!(sys)
    E_new_total = energy(sys)

    if accept!(alg, E_new_total, E_old_total)
        return nothing
    else
        sys.positions[idx] = old_pos
        _recompute_energy!(sys)
        return nothing
    end
end

"""
    _monomer_energy(sys, idx, pos) -> T

Compute the total energy contribution of monomer at `idx` with position `pos`.
Includes pair, bond, and bending terms.
"""
function _monomer_energy(sys::BeadSpringPolymer{T}, idx::Int, pos::SVector{3,T}) where T
    E = zero(T)
    m = (idx - 1) ÷ sys.N + 1  # polymer index
    k = (idx - 1) % sys.N + 1  # monomer index within polymer

    # Pair interactions with all non-bonded monomers
    n_total = sys.M * sys.N
    for j in 1:n_total
        j == idx && continue
        # Skip covalent neighbors for pair potential
        mj = (j - 1) ÷ sys.N + 1
        kj = (j - 1) % sys.N + 1
        if mj == m && abs(kj - k) == 1
            continue
        end
        r_sq = minimum_image_sq(pos, sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end

    # Bond with predecessor
    if k > 1
        j = _monomer_idx(m, k-1, sys.N)
        r_sq = minimum_image_sq(pos, sys.positions[j], sys.L)
        E += sys.bond_potential(r_sq)
    end

    # Bond with successor
    if k < sys.N
        j = _monomer_idx(m, k+1, sys.N)
        r_sq = minimum_image_sq(pos, sys.positions[j], sys.L)
        E += sys.bond_potential(r_sq)
    end

    # Bending at this monomer (k-1, k, k+1)
    if !(sys.bending_potential isa NoBendingPotential)
        if k > 1 && k < sys.N
            cos_theta = _cos_angle(sys.positions[_monomer_idx(m, k-1, sys.N)],
                                    pos,
                                    sys.positions[_monomer_idx(m, k+1, sys.N)], sys.L)
            E += sys.bending_potential(cos_theta)
        end
        # Bending at predecessor (k-2, k-1, k)
        if k > 2
            cos_theta = _cos_angle(sys.positions[_monomer_idx(m, k-2, sys.N)],
                                    sys.positions[_monomer_idx(m, k-1, sys.N)],
                                    pos, sys.L)
            E += sys.bending_potential(cos_theta)
        end
        # Bending at successor (k, k+1, k+2)
        if k < sys.N - 1
            cos_theta = _cos_angle(pos,
                                    sys.positions[_monomer_idx(m, k+1, sys.N)],
                                    sys.positions[_monomer_idx(m, k+2, sys.N)], sys.L)
            E += sys.bending_potential(cos_theta)
        end
    end

    return E
end
