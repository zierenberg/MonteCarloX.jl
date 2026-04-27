"""
    CosineBendingPotential{T} <: AbstractBendingPotential

Cosine bending potential for semiflexible chains:

    V(θ) = κ (1 - cos θ)

where κ is the bending stiffness. Evaluated as a function of cos(θ).
"""
struct CosineBendingPotential{T<:AbstractFloat} <: AbstractBendingPotential
    kappa::T
end

@inline (pot::CosineBendingPotential)(cos_theta) = pot.kappa * (1 - cos_theta)
