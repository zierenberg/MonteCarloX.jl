# Importance Sampling - Core functionality
# Shared by all importance sampling algorithms (Metropolis, Heat Bath, etc.)
"""
    ImportanceSampling <: AbstractImportanceSampling

Generic importance-sampling algorithm that operates on full
acceptance arguments `(arg_new, arg_old)` using a callable `ensemble`.

The callable may be a function or a log-weight object and should return a
scalar score such as a log density / log weight.

Conceptual API expectations:
- `ensemble(alg)` returns the architectural ensemble object carried by the algorithm.
- `logweight(alg)` returns a callable score object/function derived from that ensemble.
- Acceptance logic uses score differences from `logweight`.

Unified view:
- Bayesian inference: `logweight(theta) = logposterior(theta)`
- Statistical mechanics: `logweight(x) = -beta * E(x)`

Both are represented identically as ensemble-provided logweight callables.
"""
mutable struct ImportanceSampling{LW,RNG<:AbstractRNG} <: AbstractImportanceSampling
    rng::RNG
    ensemble::LW
    steps::Int
    accepted::Int
end

ImportanceSampling(rng::AbstractRNG, ensemble) = ImportanceSampling(rng, _as_ensemble(ensemble), 0, 0)

"""
    ensemble(alg::AbstractImportanceSampling)

Return the ensemble object carried by an importance-sampling algorithm.

This is the canonical accessor in the ensemble-first API.
Operationally, this object defines the logweight used in acceptance.
"""
@inline ensemble(alg::AbstractImportanceSampling) = getfield(alg, :ensemble)

"""
    logweight(alg::AbstractImportanceSampling)

Return the algorithm ensemble via a logweight-oriented alias.
Equivalent to `ensemble(alg)`.

Use this accessor when reasoning about acceptance formulas.
"""
@inline logweight(alg::AbstractImportanceSampling) = logweight(ensemble(alg))

# Optional ensemble-level visit hooks used by generic accept!.
# Ensembles that need histogram/visit bookkeeping can specialize these.
@inline should_record_visit(ens) = false
@inline record_visit!(_ens, _arg_vis) = nothing

"""
    accept!(alg::AbstractImportanceSampling, arg_new, arg_old) -> Bool

Evaluate the Metropolis acceptance criterion for the proposed move.

`arg_new` and `arg_old` are the arguments at which the ensemble's `logweight`
is evaluated for the proposed and current state — *not* necessarily the
system's full configuration:

  - `BoltzmannEnsemble(β)`            → energies (scalars)
  - `FunctionEnsemble(logposterior)`  → parameter vectors θ
  - `MulticanonicalEnsemble(bins)`    → values of the binned reaction coordinate

This decoupling lets adaptive ensembles operate on projected coordinates
without materializing or comparing full states.

Updates step and acceptance counters. Returns `true` if the move is accepted:
- accept if `log_ratio > 0` (proposed argument has higher weight),
- otherwise accept with probability `exp(log_ratio)`.
"""
@inline function accept!(alg::AbstractImportanceSampling, arg_new::T, arg_old::T) where T
    ens = ensemble(alg)
    log_ratio = logweight(ens, arg_new) - logweight(ens, arg_old)
    accepted = _accept!(alg, log_ratio)
    if should_record_visit(ens)
        arg_vis = accepted ? arg_new : arg_old
        record_visit!(ens, arg_vis)
    end
    return accepted
end
# core function to evaluate acceptance and update counters
@inline function _accept!(alg::AbstractImportanceSampling, log_ratio::Real)
    alg.steps += 1
    accepted = (log_ratio > 0) || (rand(alg.rng) < exp(log_ratio))
    alg.accepted += accepted 
    return accepted
end

"""
    acceptance_rate(alg::AbstractImportanceSampling)

Calculate the acceptance rate of the algorithm.

Returns the fraction of accepted moves: accepted/steps.
Returns 0.0 if no steps have been attempted yet.
"""
acceptance_rate(alg::AbstractImportanceSampling) = 
    alg.steps > 0 ? alg.accepted / alg.steps : 0.0

"""
    reset!(alg::AbstractImportanceSampling)

Reset step and acceptance counters to zero.

Useful when you want to measure acceptance rate for a specific
run phase without previous history.
"""
@inline reset!(alg::AbstractImportanceSampling) = _reset!(alg)
function _reset!(alg::AbstractImportanceSampling)
    alg.steps = 0
    alg.accepted = 0
end
