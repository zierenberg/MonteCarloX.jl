using Random

"""
    HeatBathAlgorithm{E,RNG} <: AbstractMarkovChainMonteCarlo

    HeatBathAlgorithm(rng, ensemble)
    HeatBathAlgorithm(rng; β)

Heat-bath (single-site Gibbs) sampler: draw the new local state directly from the conditional
`∝ exp(logweight(ensemble, E(s')))` over the local states. There is no accept/reject step and
hence no `accepted` counter — which is why this is a sibling of the accept/reject engine
[`MetropolisHastingsAlgorithm`](@ref) under [`AbstractMarkovChainMonteCarlo`](@ref), not an
instance of it. For a two-state update the conditional coincides with [`GlauberBalance`](@ref).

The concrete conditional lives with the system (the companion packages enumerate their local
states); carrying an `ensemble` rather than a raw `β` keeps that update one β-free expression.
"""
mutable struct HeatBathAlgorithm{E,RNG<:AbstractRNG} <: AbstractMarkovChainMonteCarlo
    rng::RNG
    ensemble::E
    steps::Int
end

HeatBathAlgorithm(rng::AbstractRNG, ensemble) = HeatBathAlgorithm(rng, _as_ensemble(ensemble), 0)
HeatBathAlgorithm(rng::AbstractRNG; β::Real) = HeatBathAlgorithm(rng, BoltzmannEnsemble(β=β), 0)
