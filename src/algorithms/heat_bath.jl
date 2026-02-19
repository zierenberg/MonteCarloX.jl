using Random

"""
    HeatBath <: AbstractHeatBath

Heat-bath sampler parameters.

For Ising-like systems, updates draw directly from the local conditional
probability using inverse temperature `β`.
"""
mutable struct HeatBath{T<:Real,RNG<:AbstractRNG} <: AbstractHeatBath
    rng::RNG
    β::T
    steps::Int
end

HeatBath(rng::AbstractRNG; β::Real) = HeatBath(rng, β, 0)
