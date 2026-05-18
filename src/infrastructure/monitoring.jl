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

# ── Weight correction helpers ──────────────────────────────────────────────────

"""
    extrapolate!(lw::BinnedObject, x_range::Tuple{Real,Real}; anchor::Real, slope::Real)

Set values in `[x_range[1], x_range[2]]` to a linear extrapolation:

    lw(x) = lw(anchor) + slope * (x - anchor)

Common slopes:
- Boltzmann extension: `slope = -β`
- From neighbors: `slope = (lw(b) - lw(a)) / (b - a)`
"""
function extrapolate!(lw::BinnedObject, x_range::Tuple{Real,Real}; anchor::Real, slope::Real)
    cs = get_centers(lw, 1)
    n = length(cs)
    x_lo, x_hi = minmax(x_range[1], x_range[2])
    idx_left = clamp(searchsortedfirst(cs, x_lo), 1, n)
    idx_right = clamp(searchsortedlast(cs, x_hi), 1, n)
    anchor_val = lw(anchor)
    @inbounds for i in idx_left:idx_right
        lw.values[i] = anchor_val + slope * (cs[i] - Float64(anchor))
    end
    return nothing
end

"""
    interpolate_gaps!(lw::BinnedObject, histogram::BinnedObject, x_range::Tuple{Real,Real})

Fill bins with zero histogram entries by linear interpolation between
nearest occupied neighbors. Leaves occupied bins unchanged.
"""
function interpolate_gaps!(lw::BinnedObject, histogram::BinnedObject, x_range::Tuple{Real,Real})
    cs = get_centers(lw, 1)
    n = length(cs)
    x_lo, x_hi = minmax(x_range[1], x_range[2])
    idx_left = clamp(searchsortedfirst(cs, x_lo), 1, n)
    idx_right = clamp(searchsortedlast(cs, x_hi), 1, n)

    gap_start = 0
    @inbounds for i in idx_left:idx_right
        if histogram.values[i] <= 0 && gap_start == 0
            gap_start = i
        elseif histogram.values[i] > 0 && gap_start > 0
            # found end of gap — interpolate if left neighbor exists
            gap_end = i - 1
            left = gap_start - 1
            right = i
            if left >= idx_left
                lw_left = lw.values[left]
                lw_right = lw.values[right]
                span = right - left
                for j in gap_start:gap_end
                    t = (j - left) / span
                    lw.values[j] = lw_left + t * (lw_right - lw_left)
                end
            end
            gap_start = 0
        end
    end
    return nothing
end

"""
    smooth!(lw::BinnedObject, x_range::Tuple{Real,Real}; window::Int=3)

Apply a rectangular (moving-average) filter of width `window` to values
in `[x_range[1], x_range[2]]`.
"""
function smooth!(lw::BinnedObject, x_range::Tuple{Real,Real}; window::Int=3)
    window >= 1 || throw(ArgumentError("window must be >= 1"))
    cs = get_centers(lw, 1)
    n = length(cs)
    x_lo, x_hi = minmax(x_range[1], x_range[2])
    idx_left = clamp(searchsortedfirst(cs, x_lo), 1, n)
    idx_right = clamp(searchsortedlast(cs, x_hi), 1, n)

    hw = div(window, 2)
    original = copy(lw.values)
    @inbounds for i in idx_left:idx_right
        lo = max(i - hw, 1)
        hi = min(i + hw, n)
        s = 0.0
        for j in lo:hi
            s += original[j]
        end
        lw.values[i] = s / (hi - lo + 1)
    end
    return nothing
end