"""
    ParticleGas{D, T, TPair} <: AbstractSoftMatterSystem

D-dimensional particle gas in a cubic box with periodic boundary conditions.

# Constructor
    ParticleGas(; D=3, N, L, pair_potential, delta=0.1)
    ParticleGas(; D=3, N, rho, pair_potential, delta=0.1)   # from density
"""
mutable struct ParticleGas{D, T<:AbstractFloat, TPair<:AbstractPairPotential} <: AbstractSoftMatterSystem
    positions::Vector{SVector{D,T}}
    N::Int
    L::T
    pair_potential::TPair
    delta::T
    cached_energy::T
end

# ── Constructors ─────────────────────────────────────────────────────────────

function ParticleGas(; D::Int=3,
                       N::Integer,
                       L=nothing,
                       rho=nothing,
                       pair_potential::AbstractPairPotential,
                       delta=0.1)
    @assert (L !== nothing) ⊻ (rho !== nothing) "Provide either `L` or `rho`, not both"
    if rho !== nothing
        L = (N / rho)^(1/D)
    end
    T = promote_type(typeof(float(L)), typeof(float(delta)))
    positions = [zero(SVector{D,T}) for _ in 1:N]
    ParticleGas{D, T, typeof(pair_potential)}(
        positions, Int(N), T(L), pair_potential, T(delta), zero(T))
end

# ── Accessors ────────────────────────────────────────────────────────────────

num_particles(sys::ParticleGas) = sys.N

# ── Initialization ───────────────────────────────────────────────────────────

function init!(sys::ParticleGas{D,T}, type::Symbol; rng=nothing) where {D,T}
    if type == :random
        @assert rng !== nothing "Random initialization requires rng"
        for i in 1:sys.N
            sys.positions[i] = SVector{D,T}(ntuple(_ -> rand(rng, T) * sys.L, Val(D)))
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_energy!(sys)
    return sys
end

# ── Energy ───────────────────────────────────────────────────────────────────

function _recompute_energy!(sys::ParticleGas{D,T}) where {D,T}
    E = zero(T)
    @inbounds for i in 1:sys.N-1
        for j in i+1:sys.N
            r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
            E += sys.pair_potential(r_sq)
        end
    end
    sys.cached_energy = E
    return nothing
end

function energy(sys::ParticleGas; full::Bool=false)
    full && _recompute_energy!(sys)
    return sys.cached_energy
end

energy_pair(sys::ParticleGas) = energy(sys; full=true)

"""
    _energy_of_particle(sys, i) -> T

Sum of pair interactions between particle i and all other particles.
"""
@inline function _energy_of_particle(sys::ParticleGas{D,T}, i::Int) where {D,T}
    E = zero(T)
    @inbounds for j in 1:sys.N
        j == i && continue
        r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end
