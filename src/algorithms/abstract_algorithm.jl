"""
    AbstractAlgorithm

Base type for all Monte Carlo algorithms.
"""
abstract type AbstractAlgorithm end

"""
    AbstractMarkovChainMonteCarlo <: AbstractAlgorithm

Umbrella type for all Markov-chain Monte Carlo algorithms. This is the whole category, so it
covers both readings of a Markov chain:

- the accept/reject engine [`MetropolisHastingsAlgorithm`](@ref) (Metropolis, Glauber, Multicanonical,
  Wang-Landau, …), which carries an `ensemble`, a `balance`, and `steps`/`accepted` counters;
- direct conditional samplers such as [`HeatBath`](@ref), which carry an `ensemble` and `steps`
  but no accept step (no `accepted` counter).

The accept/reject interface (`accept!`, `acceptance_rate`) therefore lives on the concrete
`MetropolisHastingsAlgorithm`, not here, so conditional samplers do not inherit a meaningless counter.
"""
abstract type AbstractMarkovChainMonteCarlo <: AbstractAlgorithm end

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
@inline steps(alg::AbstractKineticMonteCarlo) = getfield(alg, :steps)

"""
    ensemble(alg::AbstractMarkovChainMonteCarlo)

Return the ensemble object carried by an MCMC algorithm — the object whose `logweight` defines
the acceptance.
"""
@inline ensemble(alg::AbstractMarkovChainMonteCarlo) = getfield(alg, :ensemble)

"""
    logweight(alg::AbstractMarkovChainMonteCarlo)

Return the algorithm's ensemble as a logweight callable. Equivalent to `logweight(ensemble(alg))`.
"""
@inline logweight(alg::AbstractMarkovChainMonteCarlo) = logweight(ensemble(alg))

function Base.:(==)(a::T, b::T) where {T<:AbstractAlgorithm}
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end
