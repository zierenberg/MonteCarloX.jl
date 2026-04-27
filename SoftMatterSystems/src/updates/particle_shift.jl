"""
    particle_move!(sys::ParticleGas, alg::AbstractMetropolis)

Single-particle displacement move for a particle gas.
Picks a random particle, shifts it by a random vector in [-delta, delta]³,
and accepts/rejects via Metropolis criterion.
"""
function particle_move!(sys::ParticleGas{T}, alg::AbstractMetropolis) where T
    rng = alg.rng
    i = rand(rng, 1:sys.N)

    # Old energy contribution of particle i
    E_old_i = _energy_of_particle(sys, i)

    # Propose displacement
    old_pos = sys.positions[i]
    dx = sys.delta * (T(2) * rand(rng, T) - one(T))
    dy = sys.delta * (T(2) * rand(rng, T) - one(T))
    dz = sys.delta * (T(2) * rand(rng, T) - one(T))
    sys.positions[i] = wrap_position(old_pos + SVector{3,T}(dx, dy, dz), sys.L)

    # New energy contribution of particle i
    E_new_i = _energy_of_particle(sys, i)
    ΔE = E_new_i - E_old_i

    if accept!(alg, ΔE)
        sys.cached_energy += ΔE
    else
        sys.positions[i] = old_pos
    end
    return nothing
end

"""
    particle_move!(sys::ParticleGas, alg::AbstractImportanceSampling)

Single-particle displacement with generic importance sampling.
"""
function particle_move!(sys::ParticleGas{T}, alg::AbstractImportanceSampling) where T
    rng = alg.rng
    i = rand(rng, 1:sys.N)

    E_old = energy(sys)
    old_pos = sys.positions[i]

    dx = sys.delta * (T(2) * rand(rng, T) - one(T))
    dy = sys.delta * (T(2) * rand(rng, T) - one(T))
    dz = sys.delta * (T(2) * rand(rng, T) - one(T))
    sys.positions[i] = wrap_position(old_pos + SVector{3,T}(dx, dy, dz), sys.L)

    E_old_i = zero(T)
    E_new_i = zero(T)
    for j in 1:sys.N
        j == i && continue
        r_sq_old = minimum_image_sq(old_pos, sys.positions[j], sys.L)
        r_sq_new = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E_old_i += sys.pair_potential(r_sq_old)
        E_new_i += sys.pair_potential(r_sq_new)
    end
    E_new = E_old + E_new_i - E_old_i

    if accept!(alg, E_new, E_old)
        sys.cached_energy = E_new
    else
        sys.positions[i] = old_pos
    end
    return nothing
end
