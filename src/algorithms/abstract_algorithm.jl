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

"""
    steps(alg::AbstractAlgorithm) -> Int

Return the total number of attempted updates/events for an algorithm.
Algorithm implementations must provide this method.
"""
function steps(alg::AbstractAlgorithm)
    throw(ArgumentError("steps(::$(typeof(alg))) is not implemented; all algorithms must expose update counters"))
end

@inline steps(alg::AbstractImportanceSampling) = getfield(alg, :steps)
@inline steps(alg::AbstractHeatBath) = getfield(alg, :steps)
@inline steps(alg::AbstractKineticMonteCarlo) = getfield(alg, :steps)

function Base.:(==)(a::T, b::T) where {T<:AbstractAlgorithm}
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end
