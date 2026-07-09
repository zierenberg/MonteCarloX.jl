using Random

"""
    HeatBath{E,RNG} <: AbstractMarkovChainMonteCarlo

Heat-bath (single-site Gibbs) sampler. Draws a new state directly from the local conditional
`∝ exp(logweight(ensemble, E(s')))` over the state set — there is no accept/reject step and hence
no `accepted` counter, which is why it is a sibling of the accept/reject engine
[`MetropolisHastingsAlgorithm`](@ref) under the shared [`AbstractMarkovChainMonteCarlo`](@ref) umbrella.
For a two-state update the heat-bath conditional coincides with [`GlauberBalance`](@ref).

Carrying an `ensemble` (rather than a raw `β`) lets one generic conditional over `states_tuple`
serve every discrete model; the concrete per-model update lives with the system.
"""
mutable struct HeatBath{E,RNG<:AbstractRNG} <: AbstractMarkovChainMonteCarlo
    rng::RNG
    ensemble::E
    steps::Int
end

"""
    HeatBathAlgorithm(rng, ensemble)
    HeatBathAlgorithm(rng; β)

Build a heat-bath sampler from a callable `ensemble` (or the canonical-ensemble convenience
`BoltzmannEnsemble(β=β)`).
"""
HeatBathAlgorithm(rng::AbstractRNG, ensemble) = HeatBath(rng, _as_ensemble(ensemble), 0)
HeatBathAlgorithm(rng::AbstractRNG; β::Real) = HeatBath(rng, BoltzmannEnsemble(β=β), 0)

# Deprecated raw-β constructor (the algorithm now takes an ensemble).
function HeatBath(rng::AbstractRNG; β::Real)
    Base.depwarn("`HeatBath(rng; β)` is deprecated, use `HeatBathAlgorithm(rng; β)` " *
                 "or `HeatBathAlgorithm(rng, ensemble)`", :HeatBath)
    return HeatBathAlgorithm(rng; β=β)
end
