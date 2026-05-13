"""
    CosineBendingPotential{T} <: AbstractBendingPotential

Cosine bending potential for semiflexible chains:

    V(theta) = kappa * (1 - cos(theta))

where kappa is the bending stiffness. Evaluated as a function of cos(theta).
"""
struct CosineBendingPotential{T<:AbstractFloat} <: AbstractBendingPotential
    kappa::T
end

@inline (pot::CosineBendingPotential)(cos_theta) = pot.kappa * (1 - cos_theta)
