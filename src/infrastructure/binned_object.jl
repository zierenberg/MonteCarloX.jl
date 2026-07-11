# binned log weights for discrete and continuous variables
# (designed for histogram-based methods like multicanonical sampling and Wang-Landau)
abstract type AbstractBin end
import Base: ==

#### Boundary behavior ####
# Determines behavior when accessing a BinnedObject outside its domain.
abstract type AbstractBoundary end
struct ErrorBoundary   <: AbstractBoundary end  # throw BoundsError (default)
struct NegInfBoundary  <: AbstractBoundary end  # return -Inf  (log-weights)
struct ZeroBoundary    <: AbstractBoundary end  # return zero  (histograms)

@inline _oob_value(::ErrorBoundary,  ::Type{T}) where T = throw(BoundsError())
@inline _oob_value(::NegInfBoundary, ::Type{T}) where T = T(-Inf)
@inline _oob_value(::ZeroBoundary,   ::Type{T}) where T = zero(T)

@inline _oob_setindex!(::ErrorBoundary)    = throw(BoundsError())
@inline _oob_setindex!(::AbstractBoundary) = nothing  # no-op for permissive boundaries


### Bin types

# Discrete binning: defined by start, step, and number of bins.
# Bin centers are at start + step * (0:num-1).
struct DiscreteBinning{T<:Real} <: AbstractBin
    start :: T
    step  :: T
    num   :: Int
end

# Continuous binning (uniform): O(1) lookup.
# Defined by first edge, bin width, and number of bins.
# Edges: start + step * (0:num).  Bin i covers [edge_i, edge_{i+1}).
struct ContinuousBinning{T<:Real} <: AbstractBin
    start :: T   # first edge
    step  :: T   # bin width
    num   :: Int  # number of bins
end

# Continuous binning (arbitrary edges): O(log n) lookup.
# Bin i covers [edges[i], edges[i+1]).
struct ArbitraryContinuousBinning{T<:Real} <: AbstractBin
    edges   :: Vector{T}
    centers :: Vector{T}
end

# promote uniform → arbitrary
function ArbitraryContinuousBinning(b::ContinuousBinning{T}) where T
    edges = collect(range(b.start, step=b.step, length=b.num+1))
    centers = collect(range(b.start + b.step/2, step=b.step, length=b.num))
    return ArbitraryContinuousBinning(edges, centers)
end


#### Unchecked bin index ####

@inline function _binindex(bins::NTuple{N,B}, xs::NTuple{N,Real}) where {N,B<:AbstractBin}
    ntuple(i -> _binindex(bins[i], xs[i]), N)
end
@inline _binindices(bins::NTuple{N,B}, xs::Vararg{Real,N}) where {N,B<:AbstractBin} = _binindex(bins, xs)

# DiscreteBinning
@inline function _binindex(b::DiscreteBinning, x)
    @inbounds Int(round((x - b.start) / b.step)) + 1
end
# special case for integer
@inline function _binindex(b::DiscreteBinning{T}, x::T) where T<:Integer
    div(x - b.start, b.step) + 1
end

# ContinuousBinning (uniform) — O(1)
# clamp before floor() to avoid InexactError when x is far out of range
@inline function _binindex(b::ContinuousBinning, x::Real)
    raw = (x - b.start) / b.step
    clamped = clamp(raw, -1.0, b.num + 1.0)
    floor(Int, clamped) + 1
end

# ArbitraryContinuousBinning — O(log n)
@inline function _binindex(b::ArbitraryContinuousBinning, x::Real)
    searchsortedlast(b.edges, x)
end


#### Bounds checking ####

@inline _nbins(b::DiscreteBinning) = b.num
@inline _nbins(b::ContinuousBinning) = b.num
@inline _nbins(b::ArbitraryContinuousBinning) = length(b.centers)

@inline _is_inbounds(b::AbstractBin, idx::Int) = 1 <= idx <= _nbins(b)

@inline function _checked_binindex(b::AbstractBin, x)
    idx = _binindex(b, x)
    _is_inbounds(b, idx) ? idx : nothing
end

@inline function _checked_binindices(bins::NTuple{N,B}, xs::NTuple{N,Real}) where {N,B<:AbstractBin}
    idxs = ntuple(i -> _binindex(bins[i], xs[i]), Val(N))
    oob = ntuple(i -> _is_inbounds(bins[i], idxs[i]), Val(N))
    all(oob) || return nothing
    return idxs
end


#### Bin accessors ####

# DiscreteBinning
@inline get_centers(b::DiscreteBinning) = collect(b.start:b.step:b.start + b.step*(b.num-1))
@inline get_edges(b::DiscreteBinning) = collect(b.start - b.step/2 : b.step : b.start + b.step*(b.num-1) + b.step/2)
==(a::DiscreteBinning, b::DiscreteBinning) = (a.start == b.start && a.step == b.step && a.num == b.num)

# ContinuousBinning (uniform)
@inline get_centers(b::ContinuousBinning) = collect(range(b.start + b.step/2, step=b.step, length=b.num))
@inline get_edges(b::ContinuousBinning) = collect(range(b.start, step=b.step, length=b.num+1))
==(a::ContinuousBinning, b::ContinuousBinning) = (a.start == b.start && a.step == b.step && a.num == b.num)

# ArbitraryContinuousBinning
@inline get_centers(b::ArbitraryContinuousBinning) = b.centers
@inline get_edges(b::ArbitraryContinuousBinning) = b.edges
==(a::ArbitraryContinuousBinning, b::ArbitraryContinuousBinning) = (a.edges == b.edges && a.centers == b.centers)


#### Factory functions ####

@inline function _discrete_binning_from_domain(d::AbstractRange{T}) where {T<:Real}
    return DiscreteBinning(first(d), T(step(d)), length(d))
end

@inline function _discrete_binning_from_domain(d::AbstractVector{T}) where {T<:Real}
    n = length(d)
    n >= 2 || throw(ArgumentError("Cannot create bins from a single value."))
    steps = diff(d)
    all(steps .== steps[1]) ||
        throw(ArgumentError("Non-equidistant discrete bins not supported without ExplicitBinning."))
    return DiscreteBinning(d[1], steps[1], n)
end

@inline function _continuous_binning_from_domain(d::AbstractRange{T}) where {T<:Real}
    length(d) >= 2 || throw(ArgumentError("Continuous bin edges must contain at least two values."))
    s  = T <: Integer ? Float64(first(d)) : first(d)
    st = T <: Integer ? Float64(step(d))  : step(d)
    return ContinuousBinning(s, st, length(d) - 1)
end

@inline function _continuous_binning_from_domain(d::AbstractVector{T}) where {T<:Real}
    length(d) >= 2 || throw(ArgumentError("Continuous bin edges must contain at least two values."))
    if eltype(d) <: Integer
        d = float.(d)
    end
    edges = collect(d)
    steps = diff(edges)
    # uniform spacing → ContinuousBinning (O(1)), otherwise ArbitraryContinuousBinning (O(log n))
    if all(s -> isapprox(s, steps[1]), steps)
        return ContinuousBinning(edges[1], steps[1], length(edges) - 1)
    else
        centers = @inbounds (edges[1:end-1] .+ edges[2:end]) .* 0.5
        return ArbitraryContinuousBinning(edges, centers)
    end
end

@inline function _bin_from_domain(d::Union{AbstractRange{T},AbstractVector{T}}, interpretation::Symbol) where {T<:Real}
    if interpretation === :auto
        return eltype(d) <: Integer ? _discrete_binning_from_domain(d) : _continuous_binning_from_domain(d)
    elseif interpretation === :discrete
        return _discrete_binning_from_domain(d)
    elseif interpretation === :continuous
        return _continuous_binning_from_domain(d)
    else
        throw(ArgumentError("Invalid interpretation=$(interpretation). Use :auto, :discrete, or :continuous."))
    end
end


#### BinnedObject ####

"""
    BinnedObject(domain, init; boundary=ErrorBoundary(), interpretation=:auto)

Construct a binned object for the given domain and initial value.

The `boundary` keyword controls out-of-bounds access:
- `ErrorBoundary()` (default): throw `BoundsError`
- `NegInfBoundary()`: return `-Inf` (useful for log-weights)
- `ZeroBoundary()`: return `zero(T)` (useful for histograms)

# Examples
```julia
# Discrete 1D
bo1d = BinnedObject(0:10, 0.0)
# Discrete ND
bo2d = BinnedObject((0:5, 0:5), 0.0)
# Continuous 1D (uniform edges → O(1) lookup)
bo1d_cont = BinnedObject(0.0:0.5:5.0, 0.0)
# Continuous ND
bo2d_cont = BinnedObject((0.0:0.5:5.0, 0.0:0.5:5.0), 0.0)
# Log-weight that returns -Inf outside domain
lw = BinnedObject(0:10, 0.0; boundary=NegInfBoundary())
```
"""
struct BinnedObject{N,T<:Real,B<:AbstractBin,Bnd<:AbstractBoundary}
    bins :: NTuple{N,B}
    values :: Array{T,N}
end

function BinnedObject(domain::AbstractRange{T}, init::S; boundary::AbstractBoundary=ErrorBoundary(), interpretation::Symbol=:auto) where {T<:Real, S}
    return BinnedObject((domain,), init; boundary=boundary, interpretation=interpretation)
end

function BinnedObject(domain::AbstractVector{T}, init::S; boundary::AbstractBoundary=ErrorBoundary(), interpretation::Symbol=:auto) where {T<:Real, S}
    return BinnedObject((domain,), init; boundary=boundary, interpretation=interpretation)
end

function BinnedObject(
    domains::NTuple{N,Union{AbstractRange{T},AbstractVector{T}}},
    init::Real;
    boundary::AbstractBoundary=ErrorBoundary(),
    interpretation::Symbol=:auto,
) where {N,T<:Real}
    bins = ntuple(i -> _bin_from_domain(domains[i], interpretation), N)
    # Check all bins are of the same type
    @assert all(map(b -> typeof(b) == typeof(bins[1]), bins)) "All bins must be of the same type for NTuple type stability."
    sizes = ntuple(i -> _nbins(bins[i]), N)
    values = fill(init, sizes...)
    return BinnedObject{N,typeof(init),typeof(bins[1]),typeof(boundary)}(bins, values)
end
# catch for invalid domain types
function BinnedObject(domain, init)
    throw(ArgumentError("Invalid domain type for BinnedObject: Expected AbstractRange or AbstractVector of Real numbers, or a tuple of such (**but with identical types**)."))
end
# default constructor
@inline BinnedObject(domain; boundary::AbstractBoundary=ErrorBoundary(), interpretation::Symbol=:auto) = BinnedObject(domain, 0.0; boundary=boundary, interpretation=interpretation)

# size of the values array
@inline Base.size(lw::BinnedObject) = size(lw.values)

"""
    get_centers(bo::BinnedObject, dim::Int=1)

Return bin centers along dimension `dim`.
For discrete bins this returns the bin support values.
"""
@inline get_centers(bo::BinnedObject, dim::Int=1) = get_centers(bo.bins[dim])

"""
    get_values(bo::BinnedObject)

Return the underlying array of bin values.
"""
@inline get_values(bo::BinnedObject) = bo.values

"""
    get_edges(bo::BinnedObject, dim::Int=1)
Return bin edges along dimension `dim`.
For discrete bins this returns the edges between discrete values.
"""
@inline get_edges(bo::BinnedObject, dim::Int=1) = get_edges(bo.bins[dim])

# access via lw() syntax
@inline function (lw::BinnedObject{1,T,B,Bnd})(x::Real) where {T,B,Bnd}
    idx = _checked_binindex(lw.bins[1], x)
    idx === nothing && return _oob_value(Bnd(), T)
    return @inbounds lw.values[idx]
end
@inline function (lw::BinnedObject{N,T,B,Bnd})(xs::Vararg{Real,N}) where {N,T,B,Bnd}
    idxs = _checked_binindices(lw.bins, xs)
    idxs === nothing && return _oob_value(Bnd(), T)
    return @inbounds lw.values[idxs...]
end

# access via lw[] syntax
@inline function Base.getindex(lw::BinnedObject{1,T,B,Bnd}, x::Real) where {T,B,Bnd}
    idx = _checked_binindex(lw.bins[1], x)
    idx === nothing && return _oob_value(Bnd(), T)
    return @inbounds lw.values[idx]
end
@inline function Base.getindex(lw::BinnedObject{N,T,B,Bnd}, xs::Vararg{Real,N}) where {N,T,B,Bnd}
    idxs = _checked_binindices(lw.bins, xs)
    idxs === nothing && return _oob_value(Bnd(), T)
    return @inbounds lw.values[idxs...]
end
@inline function Base.setindex!(lw::BinnedObject{1,T,B,Bnd}, v, x::Real) where {T,B,Bnd}
    idx = _checked_binindex(lw.bins[1], x)
    if idx === nothing
        _oob_setindex!(Bnd())
        return v
    end
    return @inbounds (lw.values[idx] = v)
end
@inline function Base.setindex!(lw::BinnedObject{N,T,B,Bnd}, v, xs::Vararg{Real,N}) where {N,T,B,Bnd}
    idxs = _checked_binindices(lw.bins, xs)
    if idxs === nothing
        _oob_setindex!(Bnd())
        return v
    end
    return @inbounds (lw.values[idxs...] = v)
end

# BinnedObject equality
==(a::BinnedObject, b::BinnedObject) = (a.values == b.values && a.bins == b.bins)

"""
    zero(lw::BinnedObject)

Return a new `BinnedObject` of the same bins and boundary as `lw` but with all values set to zero.
"""
function Base.zero(lw::BinnedObject{N,T,B,Bnd}) where {N,T,B,Bnd}
    new_values = fill(zero(T), size(lw.values))
    return BinnedObject{N,T,B,Bnd}(lw.bins, new_values)
end

# helper to check if two BinnedObject objects have the same binning structure
@inline function _assert_same_domain(lw1::BinnedObject, lw2::BinnedObject)
    @assert length(lw1.bins) == length(lw2.bins) "BinnedObject objects must have the same number of dimensions."
    for i in 1:length(lw1.bins)
        b1, b2 = lw1.bins[i], lw2.bins[i]
        @assert typeof(b1) == typeof(b2) "Bin types must match in each dimension."
        if b1 isa DiscreteBinning
            @assert b1.start == b2.start "Discrete bins must have the same start."
            @assert b1.step == b2.step "Discrete bins must have the same step."
            @assert b1.num == b2.num "Discrete bins must have the same num."
        end
        if b1 isa ContinuousBinning
            @assert b1.start == b2.start "Continuous bins must have the same start."
            @assert b1.step == b2.step "Continuous bins must have the same step."
            @assert b1.num == b2.num "Continuous bins must have the same num."
        end
        if b1 isa ArbitraryContinuousBinning
            @assert b1.edges == b2.edges "Arbitrary continuous bins must have the same edges."
        end
    end
    return nothing
end


function set!(
    bo::BinnedObject,
    xrange::Union{Tuple{<:Real,<:Real},AbstractRange{<:Real}},
    f::Function,
)
    length(size(bo.values)) == 1 ||
        throw(ArgumentError("`set!` currently supports only 1D binned log-weights"))

    cs = get_centers(bo, 1)
    n = length(cs)

    xleft, xright = if xrange isa Tuple
        Float64(min(xrange[1], xrange[2])), Float64(max(xrange[1], xrange[2]))
    else
        Float64(min(first(xrange), last(xrange))), Float64(max(first(xrange), last(xrange)))
    end

    idx_left = clamp(searchsortedfirst(cs, xleft), 1, n)
    idx_right = clamp(searchsortedlast(cs, xright), 1, n)
    idx_left <= idx_right ||
        throw(ArgumentError("selected range does not overlap any bin centers"))

    if cs[idx_left] > xright || cs[idx_right] < xleft
        throw(ArgumentError("selected range does not overlap any bin centers"))
    end

    @inbounds for i in idx_left:idx_right
        x = cs[i]
        bo.values[i] = Float64(f(x))
    end

    return nothing
end
@inline set!(bo::BinnedObject, f::Function) = set!(bo, (first(get_centers(bo, 1)), last(get_centers(bo, 1))), f)

"""
    set!(target::BinnedObject, source::BinnedObject; rescale_bins=1.0, rescale_values=1.0)

Set `target` values by linearly interpolating `source` at each target bin center.
Outside the source domain, constant extrapolation (boundary values) is used.
Only 1D binned objects are supported.

- `rescale_bins`: source bin centers are multiplied by this factor before
  interpolation. Useful for reusing log-weights from a system whose domain
  is in absolute coordinates (e.g. `E`): pass `rescale_bins = N_target / N_source`.
- `rescale_values`: interpolated values are multiplied by this factor.
  Useful when the log-weight magnitude scales with system size.
"""
function set!(target::BinnedObject, source::BinnedObject;
              rescale_bins::Real=1.0, rescale_values::Real=1.0)
    ndims(target.values) == 1 ||
        throw(ArgumentError("`set!` with interpolation currently supports only 1D binned objects"))
    ndims(source.values) == 1 ||
        throw(ArgumentError("`set!` with interpolation currently supports only 1D binned objects"))

    src_cs = get_centers(source, 1)
    src_vs = source.values
    n_src  = length(src_cs)
    n_src >= 2 || throw(ArgumentError("source must have at least 2 bins for interpolation"))

    rb = Float64(rescale_bins)
    rv = Float64(rescale_values)
    tgt_cs = get_centers(target, 1)

    @inbounds for i in eachindex(tgt_cs)
        x = tgt_cs[i]
        if x <= src_cs[1] * rb
            target.values[i] = src_vs[1] * rv
        elseif x >= src_cs[n_src] * rb
            target.values[i] = src_vs[n_src] * rv
        else
            j = searchsortedlast(src_cs, x / rb)
            t = (x / rb - src_cs[j]) / (src_cs[j+1] - src_cs[j])
            target.values[i] = ((1 - t) * src_vs[j] + t * src_vs[j+1]) * rv
        end
    end

    return nothing
end

# Plotting
@recipe function f(bo::BinnedObject{1})
    seriestype --> :path
    get_centers(bo, 1), bo.values
end
