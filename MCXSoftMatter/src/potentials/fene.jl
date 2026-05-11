"""
    FENEPotential{T} <: AbstractBondPotential

Finitely Extensible Nonlinear Elastic (FENE) bond potential:

    V(r) = -(K/2) R^2 ln(1 - ((r - l0)/R)^2),   |r - l0| < R
    V(r) = Inf,                                    |r - l0| >= R

where K is the spring constant, l0 is the equilibrium bond length,
and R = l_max - l0 is the maximum extension.

Evaluated as a function of r² (takes sqrt internally).
"""
struct FENEPotential{T<:AbstractFloat} <: AbstractBondPotential
    spring_constant::T
    l0::T           # equilibrium distance
    R::T            # max extension = l_max - l0
    R_sq::T         # R^2
    inv_R_sq::T     # -1/R^2
    prefactor::T    # -K/2 * R^2
end

function FENEPotential(; spring_constant=30.0, l0=0.0, l_max=1.5)
    T = promote_type(typeof(spring_constant), typeof(l0), typeof(l_max))
    K, l0_, lm = T(spring_constant), T(l0), T(l_max)
    R = lm - l0_
    R_sq = R^2
    FENEPotential{T}(K, l0_, R, R_sq, -one(T)/R_sq, T(-0.5) * K * R_sq)
end

@inline function (pot::FENEPotential)(r_sq)
    r = sqrt(r_sq)
    diff = r - pot.l0
    diff_sq = diff * diff
    diff_sq >= pot.R_sq && return Inf
    return pot.prefactor * log1p(diff_sq * pot.inv_R_sq)
end
