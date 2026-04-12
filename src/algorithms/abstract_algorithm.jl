"""
    AbstractAlgorithm

Base type for all Monte Carlo algorithms.
"""
abstract type AbstractAlgorithm end

"""
    AbstractImportanceSampling <: AbstractAlgorithm

Base type for importance-sampling algorithms.
"""
abstract type AbstractImportanceSampling <: AbstractAlgorithm end

"""
    AbstractHeatBath <: AbstractAlgorithm

Base type for heat-bath style algorithms.
"""
abstract type AbstractHeatBath <: AbstractAlgorithm end

"""
    AbstractKineticMonteCarlo <: AbstractAlgorithm

Base type for continuous-time kinetic Monte Carlo algorithms.
"""
abstract type AbstractKineticMonteCarlo <: AbstractAlgorithm end

function Base.:(==)(a::T, b::T) where {T<:AbstractAlgorithm}
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end
