# Metropolis: the MetropolisHastingsAlgorithm engine with MetropolisBalance. A named constructor that keeps
# the call site short; Glauber is the same engine with a different balance (see glauber.jl).

"""
    MetropolisAlgorithm(rng, ensemble)
    MetropolisAlgorithm(rng; β)

Metropolis-dynamics sampler: [`MetropolisHastingsAlgorithm`](@ref) with [`MetropolisBalance`](@ref).

`MetropolisAlgorithm(rng, ensemble)` uses any callable ensemble score (log-weight object or
function); `MetropolisAlgorithm(rng; β)` is the canonical-ensemble convenience
(`BoltzmannEnsemble(β=β)`). Metropolis–Hastings needs no separate type — supply a proposal with a
nonzero ratio at the call site (it folds into the `logR` passed to [`accept!`](@ref)).
"""
MetropolisAlgorithm(rng::AbstractRNG, ensemble) =
    MetropolisHastingsAlgorithm(rng, ensemble, MetropolisBalance())
MetropolisAlgorithm(rng::AbstractRNG; β::Real) =
    MetropolisHastingsAlgorithm(rng, BoltzmannEnsemble(β=β), MetropolisBalance())
