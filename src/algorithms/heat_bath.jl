"""
    HeatBathAlgorithm{E,RNG} <: AbstractMarkovChainMonteCarlo

    HeatBathAlgorithm(rng, ensemble)
    HeatBathAlgorithm(rng; β)

Heat-bath (single-site Gibbs) sampler: draw the new local state directly from the exact
conditional via [`resample!`](@ref). There is no accept/reject step and hence no `accepted`
counter — which is why this is a sibling of the accept/reject engine
[`MetropolisHastingsAlgorithm`](@ref) under [`AbstractMarkovChainMonteCarlo`](@ref), not an
instance of it. For a two-state update the conditional coincides with [`GlauberBalance`](@ref).
"""
mutable struct HeatBathAlgorithm{E,RNG<:AbstractRNG} <: AbstractMarkovChainMonteCarlo
    rng::RNG
    ensemble::E
    steps::Int
end

function HeatBathAlgorithm(rng::AbstractRNG, ensemble)
    assert_linear_ensemble(ensemble, "heat bath")   # resample!'s shift trick needs linearity
    return HeatBathAlgorithm(rng, _as_ensemble(ensemble), 0)
end
HeatBathAlgorithm(rng::AbstractRNG; β::Real) = HeatBathAlgorithm(rng, BoltzmannEnsemble(β=β), 0)

"""
    resample!(alg::HeatBathAlgorithm, sys, i) -> s_new or nothing

Draw from the exact local conditional of site `i`, `p(s) ∝ exp(logweight(ensemble, E(s)))`
— the Gibbs / heat-bath primitive, generic over the site interface (`local_states`,
`delta_energy`). Like `accept!`, it only DECIDES: it returns the drawn state (or `nothing`
for a null move — the conditional re-drew the current state) and the caller's update
template applies it, e.g.

    s_new = resample!(alg, sys, i)
    s_new === nothing || modify!(sys, i, s_new)

One uniform draw per call, tower-sampled on the UNNORMALIZED weights: the current state
carries weight `exp(0) = 1` (energies are measured relative to it — a shift that cancels in
the conditional for LINEAR ensembles, gated at construction), alternative `s` carries
`exp(logweight(ens, ΔE_s))`; `r ~ U[0, 1 + Σw)` picks the segment. The weights are exactly
the self-normalized importance weights of the local states under the ensemble — the same
quantity `reweight` computes globally.

There is no accept/reject step, but the conditional may re-draw the current state (a null
move) — heat bath is *rejectionless*, not rejection-free. For the rejection-free
construction, where null moves are integrated into waiting times, see [`NFoldRates`](@ref).

Works on argument DIFFERENCES and therefore requires a linear ensemble (gated at
construction). Planned extension: a variant taking an explicit list of conditional weights,
for targets where the weights do not derive from ΔE alone (e.g. grand-canonical
insertion/deletion).
"""
function resample!(alg::HeatBathAlgorithm, sys, i::Int)
    ens = ensemble(alg)
    sts = local_states(sys, i)
    ws = map(s -> exp(logweight(ens, delta_energy(sys, i, s))), sts)
    alg.steps += 1
    r = rand(alg.rng) * (1.0 + sum(ws))     # the current state carries weight exp(0) = 1
    r <= 1.0 && return nothing               # null move: keep the current state
    cumulated_weight = 1.0                   # cumulative UNNORMALIZED weight (tower sampling,
    for a in eachindex(sts)                  # cf. the cumulated_rates scan in next_event)
        cumulated_weight += ws[a]
        if r < cumulated_weight
            return sts[a]
        end
    end
    return nothing
end
