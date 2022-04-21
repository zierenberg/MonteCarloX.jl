# Reweighting functions
# TODO: change names so that reweight is included because no longer namespace Reweighting

# new API [in development; may make things a lot more flexible]
# function reweighting_weights(list_args, log_P_source, log_P_target)
#    N = length(list_obs)
#    log_weight_diff(i) = log_P_target(list_args[i]...) - log_P_source(list_args[i]...)
#    for i in 1:N
#        log_norm = MonteCarloX.log_sum(log_norm, log_weight_diff(i))
#    end
# end
#
# function reweighting_weights!(list_args, log_P_source, log_P_target)
# end
#
# function reweight_expectation_value(list_obs::Vector{T}, reweighting_weights::ProbabilityWeights)::T where T<:Real
#    ev = zero(T)
#    for (obs, weight) in zip(list_obs, reweighting_weights)
#        ev += obs*weight
#    end
#    return ev
# end
#
# function reweight_expectation_value(f_args::Function, list_args::Vector{NTuple{N,T}}, reweighting_weights::ProbabilityWeights)::T where T<:Real
#    ev = zero(T)
#    for (args, weight) in zip(list_args, reweighting_weights)
#        ev += f_args(args)*weight
#    end
#    return ev
# end
#
##todo: is this a "thing"?
# function reweight_expectation_value(f_args::Function, reweighting_distribution::Histogram{T,N})::T where {T<:Real,N}
#    ev = zero(T)
#    for args in CartesianIndices(reweighting_distribution.edges)
#        ev += f_args(args)*reweighting_distribution[args]
#    end
#    return ev
# end
#
# function reweight_expectation_value(hist_obs::Histogram{T,N}, reweighting_distribution::Histogram{T,N})::T where {T<:Real,N}
#    ev = zero(T)
#    for args in CartesianIndices(reweighting_distribution.edges)
#        ev += hist_obs[args]*reweighting_distribution[args]
#    end
#    return ev
# end
#
##TODO: everywhere - get types semi-ok
# function reweight_distribution(list_args::Vector{NTuple{N,T}}, reweighting_weights::ProbabilityWeights, range)::Histogram{T,N} where {T<:Real, N}
#    hist = fit(Histogram, list_args, reweighting_weights, range, closed=:left)
#    return normalize(hist, mode=:pdf)
# end
#
##TODO: everywhere - gettypes semi-ok
# function reweighting_distribution(hist::Histogram, log_P_source::Function, log_P_target::Function)::Histogram
#    reweighting_distribution = zero(hist)
#    log_norm = log_normalization(log_P_target, log_P_source, hist)
#    #TODO: do iteration correct with CartesianIndices
#    for index in CartesianIndices(hist.edges)
#        args = hist.edges[index]
#        reweighting_distribution[args] = hist[args]*exp(log_P_target(args...) - log_P_source(args...)-log_norm)
#    end
#    return reweighting_distribution
# end
#
# function reweighting_distribution!(hist::Histogram{T,N}, log_P_source::Function, log_P_target::Function)::Histogram{T,N} where {T<:Real, N}
#    log_norm = log_normalization(log_P_target, log_P_source, hist)
#    for index in CartesianIndices(hist.edges)
#        args = hist.edges[index]
#        hist.weights[index] *= exp(log_P_target(args...) - log_P_source(args...)-log_norm)
#    end
# end

@doc raw"""
    expectation_value_from_timeseries(log_P_target::Function, log_P_source::Function, list_args, list_obs::Vector{Tin})::Tout where {Tin<:Number,Tout<:AbstractFloat}

Calculate the expectation value of an observable in `P_target` from a list of measured observables in `P_source`.

This can be used for observables measured in equilibrium, e.g., from methods:
- metropolis         (for each temperature separate)
- parallel tempering (for each temperature separate)
- multicanonical
- population annealing

# Background

Definition of reweighting in general:

```math
\langle O\rangle = \sum O_i P_\mathrm{target}(args_i)/P_\mathrm{source}(args_i) / \sum P_\mathrm{target}(args_i)/P_\mathrm{source}(args_i)
```

Definition of reweighting for the canonical ensemble:

```math
\langle O\rangle = \sum O_i e^{\beta_\mathrm{target} E_i - \beta_\mathrm{source} E_i} / \sum e^{\beta_\mathrm{target} E_i - \beta_\mathrm{source} E_i}
```

# Remark
So far, this may not be well implemented for type stability. However, it should
not be the most timeconsuming part of the simulation so this problem is moved
to later time.
"""
function expectation_value_from_timeseries(log_P_target::Function, log_P_source::Function, list_args, list_obs::Vector{Tin})::Float64 where {Tin <: Number}
    N = length(list_obs)
    @assert N == length(list_args)
    # function for difference between logarithmic weights instead of copying this
    # into an extra array. Unclear if this is better.
    log_weight_diff(i) = log_P_target(list_args[i]...) - log_P_source(list_args[i]...)

    log_norm = -Inf
    for i in 1:N
        log_norm = MonteCarloX.log_sum(log_norm, log_weight_diff(i))
    end

    expectation_value = 0
    for i in 1:N
        expectation_value += list_obs[i] * exp(log_weight_diff(i) - log_norm)
    end
    return expectation_value
end


@doc """
    distribution_from_timeseries(log_P_target, log_P_source, list_args, range [, mode])

estimate reweighted distribution over `range` for target weights `log_P_target`
from a timeseries `list_args` that was measured with `log_P_source`. Intended
also for n-dimensional probability distributions, e.g. P(E,M), where list_args
needs to be a list of tuples

returns:
    - StatsBase.Histogram

"""
function distribution_from_timeseries(log_P_target, log_P_source, list_args, range; mode=:pdf)
    N = length(list_args)
    list_weights = Weights([log_P_target(list_args[i]...) - log_P_source(list_args[i]...) for i = 1:N])

    log_norm = -Inf
    for i in 1:N
        log_norm = MonteCarloX.log_sum(log_norm, list_weights[i])
    end
    list_weights .= exp.(list_weights .- log_norm)
    hist = fit(Histogram, list_args, list_weights, range, closed = :left)
    return normalize(hist, mode = mode)
end


"""
Estimate expectation value from histogram (

Ref: Janke

Todo: rename log_P... to log_weight!!

important: hist_obs(args) = sum O_i delta(args - args_i)
hists are dictionaries?
can this be generalized to higher dimensions? nd histograms as dictionary?
"""
function expectation_value_from_histogram(f_args::Function, log_P_target::Function, log_P_source::Function, hist::Histogram)
    log_norm = _log_normalization(log_P_target, log_P_source, hist)
    expectation_value = 0
    for (args, H) in zip(hist.edges[1], hist.weights)
        expectation_value += f_args(args) * H * exp(log_P_target(args...) - log_P_source(args...) - log_norm)
    end
    return expectation_value
end

# TODO: this is only valid for 1D histograms!!
# what is missing in StatsBase is an API to iterate over histogram edges and weights (multidimensional)
function expectation_value_from_histogram(log_P_target::Function, log_P_source::Function, hist::Histogram, hist_obs::Histogram)
    log_norm = _log_normalization(log_P_target, log_P_source, hist)
    expectation_value = 0
    for (args, sum_obs) in zip(hist_obs.edges[1], hist_obs.weights)
        expectation_value += sum_obs * exp(log_P_target(args...) - log_P_source(args...) - log_norm)
    end
    return expectation_value
end

function _log_normalization(log_P_target, log_P_source, hist::Histogram)
    log_norm = -Inf
    for (args, H) in zip(hist.edges[1], hist.weights)
        log_norm = MonteCarloX.log_sum(log_norm, log(H) + log_P_target(args...) - log_P_source(args...))
    end
    return log_norm
end


###############################################################################
### Special wrapper for special ensembles

"""
timeseries reweighting in canonical ensemble

methods:
- metropolis    (for each temperature separate)

<O> = sum O_i P_target(E_i)/P_source(E_i) / sum P_target(E_i)/P_source(args_i)

"""

