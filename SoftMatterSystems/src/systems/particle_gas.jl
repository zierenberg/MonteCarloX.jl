"""
    ParticleGas{T, TPot} <: AbstractSoftMatterSystem

Off-lattice particle gas in a cubic box with periodic boundary conditions.
The pair interaction is parameterized by `TPot <: AbstractPairPotential`,
allowing e.g. Lennard-Jones, repulsive sphere, or custom potentials.

# Fields
- `positions::Vector{SVector{3,T}}` -- particle positions
- `N::Int` -- number of particles
- `L::T` -- box side length
- `pair_potential::TPot` -- pair interaction functor
- `delta::T` -- max displacement per component in MC moves
- `cached_energy::T` -- cached total potential energy
"""
mutable struct ParticleGas{T<:AbstractFloat, TPot<:AbstractPairPotential} <: AbstractSoftMatterSystem
    positions::Vector{SVector{3,T}}
    N::Int
    L::T
    pair_potential::TPot
    delta::T
    cached_energy::T
end

"""
    ParticleGas(N; L, pair_potential, delta=0.1)

Construct an uninitialized ParticleGas. Call `init!` to place particles.
"""
function ParticleGas(N::Int; L::T, pair_potential::TPot,
                      delta::T=T(0.1)) where {T<:AbstractFloat, TPot<:AbstractPairPotential}
    positions = [zero(SVector{3,T}) for _ in 1:N]
    ParticleGas{T,TPot}(positions, N, L, pair_potential, delta, zero(T))
end

"""
    ParticleGas(N; rho, pair_potential, delta=0.1)

Convenience constructor from number density ρ = N/L³.
"""
function ParticleGas(N::Int, rho::T; pair_potential::TPot,
                      delta::T=T(0.1)) where {T<:AbstractFloat, TPot<:AbstractPairPotential}
    L = (N / rho)^(one(T)/3)
    ParticleGas(N; L=L, pair_potential=pair_potential, delta=delta)
end

"""
    init!(sys::ParticleGas, type::Symbol; rng=nothing)

Initialize particle positions.

- `:random` -- place particles uniformly at random in the box
"""
function init!(sys::ParticleGas{T}, type::Symbol; rng=nothing) where T
    if type == :random
        @assert rng !== nothing "Random initialization requires rng"
        for i in 1:sys.N
            sys.positions[i] = SVector{3,T}(
                rand(rng, T) * sys.L,
                rand(rng, T) * sys.L,
                rand(rng, T) * sys.L
            )
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_energy!(sys)
    return sys
end

function _recompute_energy!(sys::ParticleGas{T}) where T
    E = zero(T)
    for i in 1:sys.N-1
        for j in i+1:sys.N
            r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
            E += sys.pair_potential(r_sq)
        end
    end
    sys.cached_energy = E
    return nothing
end

@inline function energy(sys::ParticleGas; full=false)
    full && _recompute_energy!(sys)
    return sys.cached_energy
end

"""
    _energy_of_particle(sys::ParticleGas, i) -> T

Sum of pair interactions between particle i and all other particles.
"""
@inline function _energy_of_particle(sys::ParticleGas{T}, i::Int) where T
    E = zero(T)
    for j in 1:sys.N
        j == i && continue
        r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end
