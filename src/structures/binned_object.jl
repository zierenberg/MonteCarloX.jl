# binned log weights for discrete and continuous variables 
# (designed for histogram-based methods like multicanonical sampling and Wang-Landau)
abstract type AbstractBin end
import Base: ==

@inline function _binindex(bins::NTuple{N,B}, xs::NTuple{N,Real}) where {N,B<:AbstractBin}
    ntuple(i -> _binindex(bins[i], xs[i]), N)
end
@inline _binindices(bins::NTuple{N,B}, xs::Vararg{Real,N}) where {N,B<:AbstractBin} = _binindex(bins, xs)


# Discrete binning: defined by start, step, and number of bins. Bin centers are at start + step * (0:num-1).
struct DiscreteBinning{T<:Real} <: AbstractBin
    start :: T
    step  :: T
    num   :: Int
end
@inline function _binindex(b::DiscreteBinning, x)
    @inbounds Int(round((x - b.start) / b.step)) + 1
end
# special case for integer
@inline function _binindex(b::DiscreteBinning{T}, x::T) where T<:Integer
    div(x - b.start, b.step) + 1
end
@inline get_centers(b::DiscreteBinning) = collect(b.start:b.step:b.start + b.step*(b.num-1))
@inline get_edges(b::DiscreteBinning) = collect(b.start - b.step/2 : b.step : b.start + b.step*(b.num-1) + b.step/2)
# DiscreteBinning equality
==(a::DiscreteBinning, b::DiscreteBinning) = (a.start == b.start && a.step == b.step && a.num == b.num)

# Continuous binning: defined by edges. Bin centers are midpoints between edges.
struct ContinuousBinning{T<:Real} <: AbstractBin
    edges :: Vector{T}
    centers :: Vector{T}
end
@inline function _binindex(b::ContinuousBinning, x::Real)
    searchsortedlast(b.edges, x)
end
@inline get_centers(b::ContinuousBinning) = b.centers
@inline get_edges(b::ContinuousBinning) = b.edges
# ContinuousBinning equality
==(a::ContinuousBinning, b::ContinuousBinning) = (a.edges == b.edges && a.centers == b.centers)


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

@inline function _continuous_binning_from_domain(d::Union{AbstractRange{T},AbstractVector{T}}) where {T<:Real}
    length(d) >= 2 || throw(ArgumentError("Continuous bin edges must contain at least two values."))
    # if d is type Integer, convert to float to avoid issues with non-integer bin centers
    if eltype(d) <: Integer
        d = float.(d)
    end
    edges = collect(d)
    centers = @inbounds (edges[1:end-1] .+ edges[2:end]) .* 0.5
    return ContinuousBinning(edges, centers)
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

"""
    BinnedObject(domain::AbstractRange, init)

Construct a binned log weight object for the given domain and initial value.
The type of `domain` determines the specific binned log weight type:

# Examples
```julia
# Discrete 1D
bo1d = BinnedObject(0:10, 0.0)
# Discrete ND
bo2d = BinnedObject((0:5, 0:5), 0.0)
# Continuous 1D
bo1d_cont = BinnedObject(0.0:0.5:5.0, 0.0)
# Continuous ND
bo2d_cont = BinnedObject((0.0:0.5:5.0, 0.0:0.5:5.0), 0.0)
``` 
"""
struct BinnedObject{N,T,B<:AbstractBin}
    values :: Array{T,N}
    bins :: NTuple{N,B}
end

function BinnedObject(domain::AbstractRange{T}, init::S; interpretation::Symbol=:auto) where {T<:Real, S}
    return BinnedObject((domain,), init; interpretation=interpretation)
end

function BinnedObject(domain::AbstractVector{T}, init::S; interpretation::Symbol=:auto) where {T<:Real, S}
    return BinnedObject((domain,), init; interpretation=interpretation)
end

function BinnedObject(
    domains::NTuple{N,Union{AbstractRange{T},AbstractVector{T}}},
    init::Real;
    interpretation::Symbol=:auto,
) where {N,T<:Real}
    bins = ntuple(i -> _bin_from_domain(domains[i], interpretation), N)
    # Check all bins are of the same type
    @assert all(map(b -> typeof(b) == typeof(bins[1]), bins)) "All bins must be of the same type for NTuple type stability."
    sizes = ntuple(i -> bins[i] isa DiscreteBinning ? bins[i].num : length(bins[i].edges)-1, N)
    values = fill(init, sizes...)
    return BinnedObject{N,typeof(init),typeof(bins[1])}(values, bins)
end
# catch for invalid domain types
function BinnedObject(domain, init)
    throw(ArgumentError("Invalid domain type for BinnedObject: Expected AbstractRange or AbstractVector of Real numbers, or a tuple of such (**but with identical types**)."))
end
# default constructor
@inline BinnedObject(domain; interpretation::Symbol=:auto) = BinnedObject(domain, 0.0; interpretation=interpretation)

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
@inline function (lw::BinnedObject{1})(x::Real)
    idx = _binindex(lw.bins[1], x)
    return lw.values[idx]
end
@inline function (lw::BinnedObject{N})(xs::Vararg{Real,N}) where {N}
    idxs = _binindices(lw.bins, xs...)
    return lw.values[idxs...]
end

# access via lw[] syntax
@inline function Base.getindex(lw::BinnedObject{1}, x::Real)
    idx = _binindex(lw.bins[1], x)
    return lw.values[idx]
end
@inline function Base.getindex(lw::BinnedObject{N}, xs::Vararg{Real,N}) where {N}
    idxs = _binindices(lw.bins, xs...)
    return lw.values[idxs...]
end
@inline function Base.setindex!(lw::BinnedObject{1}, v, x::Real)
    idx = _binindex(lw.bins[1], x)
    return (lw.values[idx] = v)
end
@inline function Base.setindex!(lw::BinnedObject{N}, v, xs::Vararg{Real,N}) where {N}
    idxs = _binindices(lw.bins, xs...)
    return (lw.values[idxs...] = v)
end

# BinnedObject equality
==(a::BinnedObject, b::BinnedObject) = (a.values == b.values && a.bins == b.bins)

"""
    zero(lw::BinnedObject)

Return a new `BinnedObject` of the same bins as `lw` but with all values set to zero.
"""
function Base.zero(lw::BinnedObject)
    T = eltype(lw.values)
    new_values = fill(zero(T), size(lw.values))
    return BinnedObject{length(lw.bins),T,typeof(lw.bins[1])}(new_values, lw.bins)
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
            @assert b1.edges == b2.edges "Continuous bins must have the same edges."
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