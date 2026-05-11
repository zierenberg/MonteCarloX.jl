"""
    LennardJonesPotential{T} <: AbstractPairPotential

Lennard-Jones 12-6 pair potential with cutoff and shift:

    V(r) = 4e[(s/r)^12 - (s/r)^6] - V(r_cutoff),  r < r_cutoff
    V(r) = 0,                                        r >= r_cutoff

Evaluated as a function of r² to avoid computing sqrt.
"""
struct LennardJonesPotential{T<:AbstractFloat} <: AbstractPairPotential
    epsilon::T
    sigma6::T       # sigma^6 (precomputed)
    epsilon4::T     # 4*epsilon (precomputed)
    r_cutoff_sq::T  # r_cutoff^2
    v_cutoff::T     # V(r_cutoff) for shift
end

function LennardJonesPotential(; epsilon=1.0, sigma=1.0, r_cutoff=2.5*sigma)
    T = promote_type(typeof(epsilon), typeof(sigma), typeof(r_cutoff))
    eps, sig, rc = T(epsilon), T(sigma), T(r_cutoff)
    sigma6 = sig^6
    epsilon4 = 4 * eps
    r_cutoff_sq = rc^2
    sixterm = sigma6 / r_cutoff_sq^3
    v_cutoff = epsilon4 * (sixterm^2 - sixterm)
    LennardJonesPotential{T}(eps, sigma6, epsilon4, r_cutoff_sq, v_cutoff)
end

@inline function (pot::LennardJonesPotential{T})(r_sq) where T
    r_sq > pot.r_cutoff_sq && return zero(T)
    sixterm = pot.sigma6 / (r_sq * r_sq * r_sq)
    return pot.epsilon4 * (sixterm * sixterm - sixterm) - pot.v_cutoff
end

@inline cutoff_sq(pot::LennardJonesPotential) = pot.r_cutoff_sq
