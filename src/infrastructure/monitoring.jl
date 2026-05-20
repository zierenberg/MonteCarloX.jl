"""
    Roundtrips(x_min, x_max)

Track completed round trips of a scalar between two boundaries.
A round trip is completed when the value reaches one boundary and
subsequently reaches the other.

Useful for monitoring convergence of multicanonical, Wang-Landau, or
parallel tempering simulations.

# Example
```julia
rt = Roundtrips(-100.0, 0.0)
for step in 1:N
    spin_flip!(sys, alg)
    update!(rt, energy(sys))
end
rt.count  # completed round trips
```
"""
mutable struct Roundtrips
    x_min::Float64
    x_max::Float64
    at_boundary::Int  # 0=neither, -1=was at min, +1=was at max
    count::Int
end

function Roundtrips(x_min::Real, x_max::Real)
    x_min < x_max || throw(ArgumentError("x_min must be less than x_max"))
    Roundtrips(Float64(x_min), Float64(x_max), 0, 0)
end

"""
    update!(rt::Roundtrips, x)

Update the round-trip counter with the current value `x`.
Increments `rt.count` when a round trip is completed.
"""
function update!(rt::Roundtrips, x::Real)
    if x <= rt.x_min
        if rt.at_boundary == 1
            rt.count += 1
        end
        rt.at_boundary = -1
    elseif x >= rt.x_max
        if rt.at_boundary == -1
            rt.count += 1
        end
        rt.at_boundary = 1
    end
    return nothing
end

function reset!(rt::Roundtrips)
    rt.at_boundary = 0
    rt.count = 0
    return nothing
end

# ── Histogram flatness ─────────────────────────────────────────────────────────

"""
    flatness(histogram::BinnedObject, x_min, x_max; criterion=:max_over_mean)

Compute flatness ratio of occupied bins in `[x_min, x_max]`.
Returns a ratio >= 1.0 where 1.0 means perfectly flat.

# Criteria
- `:max_over_mean` — `max(h) / mean(h)` over occupied bins
- `:mean_over_min` — `mean(h) / min(h)` over occupied bins
"""
function flatness(histogram::BinnedObject, x_min::Real, x_max::Real; criterion::Symbol=:max_over_mean)
    cs = get_centers(histogram, 1)
    n = length(cs)
    idx_left = clamp(searchsortedfirst(cs, x_min), 1, n)
    idx_right = clamp(searchsortedlast(cs, x_max), 1, n)
    idx_left <= idx_right || throw(ArgumentError("range does not overlap any bin centers"))

    occupied = Float64[]
    @inbounds for i in idx_left:idx_right
        h = histogram.values[i]
        if h > 0
            push!(occupied, h)
        end
    end
    isempty(occupied) && return Inf

    if criterion === :max_over_mean
        return maximum(occupied) / mean(occupied)
    elseif criterion === :mean_over_min
        return mean(occupied) / minimum(occupied)
    else
        throw(ArgumentError("unsupported criterion=$(criterion), use :max_over_mean or :mean_over_min"))
    end
end