"""
    AbstractAlgorithm

Base type for all Monte Carlo algorithms.
"""
abstract type AbstractAlgorithm end

"""
    AbstractMarkovChainMonteCarlo <: AbstractAlgorithm

Base type for accept/reject Markov chain Monte Carlo algorithms (Metropolis,
Glauber, Multicanonical, Wang-Landau, Replica exchange, …).
Carries an `ensemble`, `steps`, and `accepted` counters.
"""
abstract type AbstractMarkovChainMonteCarlo <: AbstractAlgorithm end

"""
    AbstractHeatBath <: AbstractAlgorithm

Base type for heat-bath / conditional MCMC algorithms.
Conceptually MCMC; kept as a sibling of [`AbstractMarkovChainMonteCarlo`](@ref)
because it lacks the accept/reject + ensemble interface (no `accepted` counter,
draws directly from local conditionals). A future refactor may introduce a
shared intermediate supertype.
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

@inline steps(alg::AbstractMarkovChainMonteCarlo) = getfield(alg, :steps)
@inline steps(alg::AbstractHeatBath) = getfield(alg, :steps)
@inline steps(alg::AbstractKineticMonteCarlo) = getfield(alg, :steps)

function Base.:(==)(a::T, b::T) where {T<:AbstractAlgorithm}
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end
