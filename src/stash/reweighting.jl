"""
    expectation_value_from_timeseries(log_P_target, log_P_source, list_args, list_obs)

Calculate an expectation value in a target ensemble from measurements produced
in a source ensemble using log-reweighting.
"""
function expectation_value_from_timeseries(
    log_P_target::Function,
    log_P_source::Function,
    list_args,
    list_obs::AbstractVector{Tin},
)::Float64 where {Tin <: Number}
    N = length(list_obs)
    @assert N == length(list_args)

    log_weight_diff(i) = log_P_target(list_args[i]...) - log_P_source(list_args[i]...)

    log_norm = -Inf
    for i in 1:N
        log_norm = MonteCarloX.log_sum(log_norm, log_weight_diff(i))
    end

    expectation_value = 0.0
    for i in 1:N
        expectation_value += list_obs[i] * exp(log_weight_diff(i) - log_norm)
    end

    return expectation_value
end

"""
    distribution_from_timeseries(log_P_target, log_P_source, list_args, range)

Estimate a (possibly multi-dimensional) target distribution from measured
arguments in a source ensemble.
"""
function distribution_from_timeseries(log_P_target, log_P_source, list_args, range)
    N = length(list_args)

    log_weights = Vector{Float64}(undef, N)
    for i in 1:N
        log_weights[i] = log_P_target(list_args[i]...) - log_P_source(list_args[i]...)
    end

    log_norm = -Inf
    for i in 1:N
        log_norm = MonteCarloX.log_sum(log_norm, log_weights[i])
    end

    weights = Weights(exp.(log_weights .- log_norm))
    hist = fit(Histogram, list_args, weights, range, closed = :left)
    return normalize(hist, mode = :pdf)
end

"""
    expectation_value_from_histogram(f_args, log_P_target, log_P_source, hist)

Estimate expectation value from a histogram representation of a source
distribution.
"""
function expectation_value_from_histogram(
    f_args::Function,
    log_P_target::Function,
    log_P_source::Function,
    hist::Histogram,
)
    log_norm = log_normalization(log_P_target, log_P_source, hist)
    expectation_value = 0.0
    for (args, H) in zip(hist.edges[1], hist.weights)
        expectation_value += f_args(args) * H * exp(log_P_target(args...) - log_P_source(args...) - log_norm)
    end
    return expectation_value
end

"""
    expectation_value_from_histogram(log_P_target, log_P_source, hist, hist_obs)

Estimate expectation value from histogrammed observables.
"""
function expectation_value_from_histogram(
    log_P_target::Function,
    log_P_source::Function,
    hist::Histogram,
    hist_obs::Histogram,
)
    log_norm = log_normalization(log_P_target, log_P_source, hist)
    expectation_value = 0.0
    for (args, sum_obs) in zip(hist_obs.edges[1], hist_obs.weights)
        expectation_value += sum_obs * exp(log_P_target(args...) - log_P_source(args...) - log_norm)
    end
    return expectation_value
end

"""
    log_normalization(log_P_target, log_P_source, hist)

Log-normalization factor used by histogram-based reweighting estimators.
"""
function log_normalization(log_P_target, log_P_source, hist::Histogram)
    log_norm = -Inf
    for (args, H) in zip(hist.edges[1], hist.weights)
        log_norm = MonteCarloX.log_sum(log_norm, log(H) + log_P_target(args...) - log_P_source(args...))
    end
    return log_norm
end
