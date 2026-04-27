"""
    LennardJones{T} <: AbstractPairPotential

Lennard-Jones 12-6 pair potential with cutoff and shift:

    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶] - V(r_cutoff),  r < r_cutoff
    V(r) = 0,                                       r ≥ r_cutoff

The potential is evaluated as a function of r² to avoid computing sqrt.
"""
struct LennardJonesPotential{T<:AbstractFloat} <: AbstractPairPotential
    epsilon::T
    sigma6::T       # σ⁶ (precomputed)
    epsilon4::T     # 4ε (precomputed)
    r_cutoff_sq::T  # r_cutoff²
    v_cutoff::T     # V(r_cutoff) for shift
end

function LennardJonesPotential(; epsilon::T=1.0, sigma::T=1.0,
                                  r_cutoff::T=T(2.5) * sigma) where T<:AbstractFloat
    sigma6 = sigma^6
    epsilon4 = 4 * epsilon
    r_cutoff_sq = r_cutoff^2
    sixterm = sigma6 / r_cutoff_sq^3
    v_cutoff = epsilon4 * (sixterm^2 - sixterm)
    LennardJonesPotential{T}(epsilon, sigma6, epsilon4, r_cutoff_sq, v_cutoff)
end

@inline function (pot::LennardJonesPotential)(r_sq)
    r_sq > pot.r_cutoff_sq && return 0.0
    sixterm = pot.sigma6 / (r_sq * r_sq * r_sq)
    return pot.epsilon4 * (sixterm * sixterm - sixterm) - pot.v_cutoff
end

@inline cutoff_sq(pot::LennardJonesPotential) = pot.r_cutoff_sq
