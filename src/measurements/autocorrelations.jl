"""
    integrated_autocorrelation_time(samples; max_lag=nothing, c=5.0)

Estimate the integrated autocorrelation time

```
tau_int = 1/2 + sum_{t>=1} C(t),
```

for a scalar time series `samples` where C(t) is the normalized autocorrelation function at lag t. 

Implementation notes:
- Uses a direct lag loop over centered samples.
- Truncates the sum using a self-consistent window (`lag > c * tau`).
- Also truncates if the estimated normalized autocorrelation becomes
  non-positive.

# Arguments
- `samples`: scalar time series (e.g. energy trace at fixed beta).
- `max_lag`: optional hard cap for the lag window. Must satisfy
  `1 <= max_lag <= floor(length(samples)/2)`.
- `c`: positive windowing constant for the self-consistent cutoff.

# Returns
- Estimated integrated autocorrelation time as `Float64`.
- Returns `0.5` for zero-variance input.
"""
function integrated_autocorrelation_time(samples::AbstractVector{<:Real};
                                         max_lag::Union{Nothing,Integer}=nothing,
                                         c::Real=5.0)
    n = length(samples)
    n >= 2 || throw(ArgumentError("integrated_autocorrelation_time requires at least 2 samples"))
    c > 0 || throw(ArgumentError("c must be positive"))

    lag_cap_max = n ÷ 2
    lag_cap_max >= 1 || throw(ArgumentError("need at least 2 samples to define a positive lag window"))

    lag_cap = isnothing(max_lag) ? lag_cap_max : Int(max_lag)
    1 <= lag_cap <= lag_cap_max ||
        throw(ArgumentError("max_lag must satisfy 1 <= max_lag <= floor(length(samples)/2)"))

    x = Float64.(samples)
    μ = sum(x) / n
    centered = x .- μ

    # C(0): variance-like normalization for the autocorrelation function.
    C0 = dot(centered, centered) / n

    C0 > 0 || return 0.5

    tau = 0.5
    @inbounds for lag in 1:lag_cap
        cov_lag = dot(view(centered, 1:(n - lag)), view(centered, (1 + lag):n)) / (n - lag)
        C = cov_lag / C0

        # Standard positive-sequence style truncation.
        C <= 0 && break

        tau_next = tau + C
        lag > c * tau_next && break
        tau = tau_next
    end

    return max(0.5, tau)
end

"""
    integrated_autocorrelation_times(traces; min_points=2, max_lag=nothing, c=5.0)

Compute `integrated_autocorrelation_time` for each trace in `traces`.
Entries with fewer than `min_points` samples return `NaN`.
"""
function integrated_autocorrelation_times(traces::AbstractVector{<:AbstractVector{<:Real}};
                                          min_points::Integer=2,
                                          max_lag::Union{Nothing,Integer}=nothing,
                                          c::Real=5.0)
    min_points >= 2 || throw(ArgumentError("min_points must be >= 2"))

    taus = fill(NaN, length(traces))
    @inbounds for i in eachindex(traces)
        trace = traces[i]
        n = length(trace)
        if n < min_points
            continue
        end

        local_max_lag = if isnothing(max_lag)
            nothing
        else
            min(Int(max_lag), n ÷ 2)
        end

        if !isnothing(local_max_lag) && local_max_lag < 1
            continue
        end

        taus[i] = integrated_autocorrelation_time(trace; max_lag=local_max_lag, c=c)
    end

    return taus
end

"""
    tau_int(samples; kwargs...)

Convenience wrapper for `integrated_autocorrelation_time`.
"""
function tau_int(samples::AbstractVector{<:Real}; kwargs...)
    return integrated_autocorrelation_time(samples; kwargs...)
end
