"""
    FENEPotential{T} <: AbstractBondPotential

Finitely Extensible Nonlinear Elastic (FENE) bond potential:

    V(r) = -(K/2) R² ln(1 - ((r - l₀)/R)²),   |r - l₀| < R
    V(r) = ∞,                                    |r - l₀| ≥ R

where K is the spring constant, l₀ is the equilibrium bond length,
and R = l_max - l₀ is the maximum extension.

Evaluated as a function of r² (takes sqrt internally for r).
"""
struct FENEPotential{T<:AbstractFloat} <: AbstractBondPotential
    spring_constant::T
    l0::T           # equilibrium distance
    R::T            # max extension = l_max - l0
    R_sq::T         # R²
    inv_R_sq::T     # -1/R²
    prefactor::T    # -K/2 * R²
end

function FENEPotential(; spring_constant::T=30.0, l0::T=0.0,
                         l_max::T=T(1.5)) where T<:AbstractFloat
    R = l_max - l0
    R_sq = R^2
    FENEPotential{T}(spring_constant, l0, R, R_sq, -one(T)/R_sq,
                      T(-0.5) * spring_constant * R_sq)
end

@inline function (pot::FENEPotential)(r_sq)
    r = sqrt(r_sq)
    diff = r - pot.l0
    diff_sq = diff * diff
    diff_sq >= pot.R_sq && return Inf
    return pot.prefactor * log1p(diff_sq * pot.inv_R_sq)
end
