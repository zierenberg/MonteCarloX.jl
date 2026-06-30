mutable struct MulticanonicalEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    d_logweight::Vector{Float64}     # n-1: d_logweight[k] = W[k+1] - W[k]
    log_cumweight::Vector{Float64}   # n-1: cumulative statistical weight per transition
    record_visits::Bool
    warn_overwrite::Bool
    smooth_window::Int
    visited_min::Float64
    visited_max::Float64

    function MulticanonicalEnsemble(logweight::BO, histogram::BO;
            record_visits::Bool=true, warn_overwrite::Bool=true, smooth_window::Int=0) where {BO<:BinnedObject}
        _assert_same_domain(logweight, histogram)
        n = length(logweight.values)
        d_lw = Vector{Float64}(undef, n - 1)
        @inbounds for k in 1:n-1
            d_lw[k] = logweight.values[k+1] - logweight.values[k]
        end
        log_cumweight = fill(-Inf, n - 1)
        new{BO}(logweight, histogram, d_lw, log_cumweight, record_visits, warn_overwrite, smooth_window, Inf, -Inf)
    end
end
MulticanonicalEnsemble(logweight::BO; histogram=nothing, kwargs...) where {BO<:BinnedObject} =
    MulticanonicalEnsemble(logweight, histogram === nothing ? zero(logweight) : histogram; kwargs...)

function MulticanonicalEnsemble(bins; init::Real=0.0, kwargs...)
    lw = bins isa BinnedObject ? bins : BinnedObject(bins, float(init), boundary=NegInfBoundary())
    histogram = zero(lw)
    return MulticanonicalEnsemble(lw, histogram; kwargs...)
end

@inline logweight(e::MulticanonicalEnsemble) = e.logweight # this is already a callable BinnedObject, so we can just return it
@inline logweight(e::MulticanonicalEnsemble, arg) = e.logweight(arg)

@inline should_record_visit(ens::MulticanonicalEnsemble) = ens.record_visits

"""
    visited_range(ens::MulticanonicalEnsemble)

Return `(x_min, x_max)` of the cumulative visited range across all iterations.
The range only grows — `reset!` does not shrink it.
Returns `(Inf, -Inf)` if no visits have been recorded yet.
"""
@inline visited_range(ens::MulticanonicalEnsemble) = (ens.visited_min, ens.visited_max)

@inline function record_visit!(ens::MulticanonicalEnsemble, x_vis)
    h = ens.histogram
    # multidimensional aspects should be handled by the BinnedObject indexing.
    h[x_vis] += 1
    x = Float64(x_vis)
    if x < ens.visited_min
        ens.visited_min = x
    end
    if x > ens.visited_max
        ens.visited_max = x
    end
    return nothing
end

#### d_logweight ↔ logweight synchronization ####

# Derive d_logweight from current logweight values: d_logweight[k] = W[k+1] - W[k]
function _differentiate!(e::MulticanonicalEnsemble)
    log_w = e.logweight.values
    @inbounds for k in 1:length(e.d_logweight)
        e.d_logweight[k] = log_w[k+1] - log_w[k]
    end
    return nothing
end

# Rebuild logweight from d_logweight, preserving W[1]: W[k+1] = W[k] + d_logweight[k].
# When `smooth_window > 0`, a moving-average filter is applied to a temporary copy
# of d_logweight before integration. This smooths the logweight used for sampling
# while preserving the raw statistical d_logweight values.
function _integrate!(e::MulticanonicalEnsemble)
    log_w = e.logweight.values
    d_lw = e.d_logweight
    n_dlw = length(d_lw)

    if e.smooth_window > 1 && n_dlw > 0
        hw = div(e.smooth_window, 2)
        @inbounds for k in 1:n_dlw
            lo = max(k - hw, 1)
            hi = min(k + hw, n_dlw)
            s = 0.0
            for j in lo:hi
                s += d_lw[j]
            end
            log_w[k+1] = log_w[k] + s / (hi - lo + 1)
        end
    else
        @inbounds for k in 1:n_dlw
            log_w[k+1] = log_w[k] + d_lw[k]
        end
    end
    return nothing
end

"""
    set!(ens::MulticanonicalEnsemble, args...)

Set logweight values and synchronize d_logweight.
Accepts the same arguments as `set!(::BinnedObject, ...)`.
"""
function set!(ens::MulticanonicalEnsemble, args...; kwargs...)
    set!(ens.logweight, args...; kwargs...)
    _differentiate!(ens)
    return nothing
end

# Emit a warning when overwriting d_logweight in regions where the recursive
# update has accumulated precision (i.e. the estimate was data-driven).
function _warn_overwrite_overwrite(e::MulticanonicalEnsemble, idx_range)
    e.warn_overwrite || return nothing
    @inbounds for k in idx_range
        if 1 <= k <= length(e.log_cumweight) && e.log_cumweight[k] > -Inf
            @warn "Overwriting d_logweight where recursive precision has been accumulated. " *
                  "This discards converged weight estimates."
            return nothing
        end
    end
    return nothing
end

#### Weight update ####

"""
    update!(e::MulticanonicalEnsemble; mode=:simple)

Update logweights from the current histogram.

# Modes
- `:simple` — trivial update: `\\log W_k = \\log\\frac{W_k}{H_k}` (Berg & Neuhaus, 1992).
- `:recursive` — precision-weighted recursive update of `d_logweight`
  (Berg, J. Stat. Phys. 82, 323, 1996; Janke, Physica A 254, 164, 1998;
  Zierenberg, Dissertation, Universität Leipzig, 2016, §5.4.2).

  For each local log-weight slope (weight-ratio) between adjacent bins with ``H_k > 0`` and
  ``H_{k+1} > 0``:

  1. **Estimate** the new logweight gradient analogous to the simple update:
     ```math
     \\Delta_k^{\\mathrm{est}} = \\log\\frac{W_{k+1}}{H_{k+1}} - \\log\\frac{W_k}{H_k}
     ```
  2. **Weight** this estimate with the harmonic mean of adjacent bin counts:
     ```math
     w_k = \\frac{H_k \\, H_{k+1}}{H_k + H_{k+1}}
     ```
  3. **Add** to the cumulative estimate via a weighted average:
     ```math
     \\Delta_k^{\\mathrm{new}} = \\frac{w_k^{\\mathrm{old}} \\, \\Delta_k^{\\mathrm{old}} + w_k \\, \\Delta_k^{\\mathrm{est}}}{w_k^{\\mathrm{old}} + w_k}
     ```

  The cumulative weight ``w_k^{\\mathrm{old}} + w_k`` is stored in
  `log_cumweight[k]` (in log-space) and grows across iterations.

  When `smooth_window > 0`, the actual sampling ``logweight`` may differ
  from the raw `d_logweight` because `_integrate!` applies smoothing.
  The estimate ``\\Delta_k^{\\mathrm{est}}`` uses the actual ``logweight`` to
  correctly account for this.
"""
function update!(e::MulticanonicalEnsemble; mode::Symbol=:simple)
    if mode === :simple
        _update_simple!(e)
    elseif mode === :recursive
        _update_recursive!(e)
    else
        throw(ArgumentError("unsupported mode=$(mode), use :simple or :recursive"))
    end
    return nothing
end

function _update_simple!(e::MulticanonicalEnsemble)
    @inbounds for idx in eachindex(e.histogram.values)
        h = e.histogram.values[idx]
        logh = h > 0 ? log(h) : 0.0
        e.logweight.values[idx] -= logh
    end
    _differentiate!(e)
end

# Multicanonical recursion (Berg, J. Stat. Phys. 82, 323, 1996;
# Janke, Physica A 254, 164, 1998;
# Zierenberg, Dissertation, Universität Leipzig, 2016, §5.4.2).
function _update_recursive!(e::MulticanonicalEnsemble)
    n = length(e.logweight.values)
    n >= 2 || return

    log_w    = e.logweight.values
    d_lw     = e.d_logweight
    log_cumw = e.log_cumweight
    hist     = e.histogram.values

    @inbounds for k in 1:n-1
        hist[k] > 0 && hist[k+1] > 0 || continue

        log_h_lo = log(hist[k])
        log_h_hi = log(hist[k+1])

        # new estimate: correct histogram for actual sampling weights
        #   g(E) ∝ H(E) / W(E), so d_lw_est = log(W[k+1]/H[k+1]) - log(W[k]/H[k])
        d_lw_est = (log_w[k+1] - log_h_hi) - (log_w[k] - log_h_lo)

        # statistical weight of this estimate: w = H_lo * H_hi / (H_lo + H_hi)
        log_w_new = log_h_lo + log_h_hi - log_sum(log_h_lo, log_h_hi)

        # precision-weighted average: d_lw = (w_old * d_lw_old + w_new * d_lw_est) / w_total
        log_w_old   = log_cumw[k]
        log_w_total = log_w_old == -Inf ? log_w_new : log_sum(log_w_old, log_w_new)
        log_cumw[k] = log_w_total

        if log_w_old == -Inf
            d_lw[k] = d_lw_est
        else
            d_lw[k] = exp(log_w_old - log_w_total) * d_lw[k] +
                       exp(log_w_new - log_w_total) * d_lw_est
        end
    end

    _integrate!(e)
end

#### Weight correction helpers ####

"""
    extend!(e::MulticanonicalEnsemble, direction::Symbol; anchor::Real, slope::Real, limit::Real=NaN)

Extend logweights by setting a constant d_logweight (slope) from `anchor`
toward the domain boundary (or up to `limit`).

- `direction = :low`  — fill from `limit` (or domain start) up to `anchor`
- `direction = :high` — fill from `anchor` to `limit` (or domain end)

The `slope` is the gradient of logweight: `dW/dx = slope`, so
`d_logweight[k] = slope * dx` for each transition in the range.

Common slopes:
- Boltzmann extension: `slope = -β`
- From neighbors: `slope = (logweight(b) - logweight(a)) / (b - a)`
"""
function extend!(e::MulticanonicalEnsemble, direction::Symbol; anchor::Real, slope::Real, limit::Real=NaN)
    cs = get_centers(e.logweight, 1)
    n = length(cs)
    if direction === :low
        idx_anchor = clamp(searchsortedlast(cs, Float64(anchor)), 1, n)
        idx_limit  = isnan(limit) ? 1 : clamp(searchsortedfirst(cs, Float64(limit)), 1, n)
        # d_logweight indices for transitions within [idx_limit, idx_anchor)
        k_first = max(idx_limit, 1)
        k_last  = idx_anchor - 1
    elseif direction === :high
        idx_anchor = clamp(searchsortedfirst(cs, Float64(anchor)), 1, n)
        idx_limit  = isnan(limit) ? n : clamp(searchsortedlast(cs, Float64(limit)), 1, n)
        # d_logweight indices for transitions within [idx_anchor, idx_limit)
        k_first = idx_anchor
        k_last  = idx_limit - 1
    else
        throw(ArgumentError("direction must be :low or :high, got :$(direction)"))
    end

    k_first <= k_last || return nothing

    _warn_overwrite_overwrite(e, k_first:k_last)

    # Set d_logweight[k] = slope * dx for each transition in range
    log_w = e.logweight.values
    @inbounds for k in k_first:k_last
        dx = cs[k+1] - cs[k]
        e.d_logweight[k] = slope * dx
    end

    # Local integration from anchor to preserve anchor value.
    # For :low, integrate backward from anchor; for :high, integrate forward.
    if direction === :low
        @inbounds for k in k_last:-1:k_first
            log_w[k] = log_w[k+1] - e.d_logweight[k]
        end
    else
        @inbounds for k in k_first:k_last
            log_w[k+1] = log_w[k] + e.d_logweight[k]
        end
    end
    return nothing
end

"""
    smooth!(e::MulticanonicalEnsemble, x_range::Tuple{Real,Real}; window::Int=3)

Apply a rectangular (moving-average) filter of width `window` to the
d_logweight values in `[x_range[1], x_range[2]]`.

This modifies the raw d_logweight values directly. For non-destructive
smoothing applied only to the logweight (preserving raw d_logweight),
set `smooth_window` on the ensemble instead.
"""
function smooth!(e::MulticanonicalEnsemble, x_range::Tuple{Real,Real}; window::Int=3)
    window >= 1 || throw(ArgumentError("window must be >= 1"))
    cs = get_centers(e.logweight, 1)
    n = length(cs)
    n_dlw = n - 1
    x_lo, x_hi = minmax(x_range[1], x_range[2])
    # Map x-range to d_logweight indices
    idx_left = clamp(searchsortedfirst(cs, x_lo), 1, n)
    idx_right = clamp(searchsortedlast(cs, x_hi), 1, n)
    k_left = max(idx_left, 1)
    k_right = min(idx_right - 1, n_dlw)
    k_left <= k_right || return nothing

    _warn_overwrite_overwrite(e, k_left:k_right)

    hw = div(window, 2)
    original = copy(e.d_logweight)
    @inbounds for k in k_left:k_right
        lo = max(k - hw, 1)
        hi = min(k + hw, n_dlw)
        s = 0.0
        for j in lo:hi
            s += original[j]
        end
        e.d_logweight[k] = s / (hi - lo + 1)
    end

    _integrate!(e)
    return nothing
end
