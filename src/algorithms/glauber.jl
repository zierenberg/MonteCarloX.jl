# Glauber: the MetropolisHastingsAlgorithm engine with GlauberBalance (≡ Barker). Same slots as
# Metropolis; only the balance function differs (see metropolis.jl / metropolis_hastings.jl).

"""
    GlauberAlgorithm(rng, ensemble)
    GlauberAlgorithm(rng; β)

Glauber-dynamics sampler: [`MetropolisHastingsAlgorithm`](@ref) with [`GlauberBalance`](@ref) (logistic
acceptance `1 / (1 + exp(-logR))`). Same slots as [`MetropolisAlgorithm`](@ref); only the balance
function differs. For a two-state local update this coincides with single-site heat bath.
"""
GlauberAlgorithm(rng::AbstractRNG, ensemble) =
    MetropolisHastingsAlgorithm(rng, ensemble, GlauberBalance())
GlauberAlgorithm(rng::AbstractRNG; β::Real) =
    MetropolisHastingsAlgorithm(rng, BoltzmannEnsemble(β=β), GlauberBalance())
