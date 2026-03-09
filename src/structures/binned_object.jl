# binned log weights for discrete and continuous variables 
# (designed for histogram-based methods like multicanonical sampling and Wang-Landau)

abstract type AbstractBin end
@inline function _binindex(bins::NTuple{N,AbstractBin}, xs::NTuple{N,Real}) where N
    ntuple(i -> _binindex(bins[i], xs[i]), N)
end

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
@inline Base.collect(b::DiscreteBinning) = Base.collect(b.start:b.step:b.start + b.step*(b.num-1))

struct ContinuousBinning{T<:Real} <: AbstractBin
    edges :: Vector{T}
    centers :: Vector{T}
end
@inline function _binindex(b::ContinuousBinning, x::Real)
    searchsortedlast(b.edges, x)
end
@inline Base.collect(b::ContinuousBinning) = b.centers

# struct ExplicitBinning{B}
#     labels :: B
# end

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

@inline function Base.getproperty(bo::BinnedObject, name::Symbol)
    if name === :weights
        return getfield(bo, :values)
    end
    return getfield(bo, name)
end

@inline function Base.setproperty!(bo::BinnedObject, name::Symbol, value)
    if name === :weights
        return setfield!(bo, :values, value)
    end
    return setfield!(bo, name, value)
end

function BinnedObject(domain::AbstractRange{T}, init::S) where {T<:Real, S}
    return BinnedObject((domain,), init)
end

function BinnedObject(domain::AbstractVector{T}, init::S) where {T<:Real, S}
    return BinnedObject((domain,), init)
end

function BinnedObject(domains::NTuple{N,Union{AbstractRange{T},AbstractVector{T}}}, init::Real) where {N,T<:Real}
    bins = ntuple(i -> begin
        d = domains[i]
        if eltype(d) <: Integer
            if d isa AbstractRange
                DiscreteBinning(first(d), T(step(d)), length(d))
            elseif d isa AbstractVector
                n = length(d)
                if n == 1
                    throw(ArgumentError("Cannot create bins from a single value."))
                else
                    steps = diff(d)
                    if all(steps .== steps[1])
                        DiscreteBinning(d[1], steps[1], n)
                    else
                        throw(ArgumentError("Non-equidistant discrete bins not supported without ExplicitBinning."))
                    end
                end
            end
        else
            edges = collect(d)
            centers = @inbounds (edges[1:end-1] .+ edges[2:end]) .* T(0.5)
            ContinuousBinning(edges, centers)
        end
    end, N)
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
@inline BinnedObject(domain) = BinnedObject(domain, 0.0)

# size of the weights array
@inline Base.size(lw::BinnedObject) = size(lw.values)

"""
    get_centers(bo::BinnedObject, dim::Int=1)

Return bin centers along dimension `dim`.
For discrete bins this returns the bin support values.
"""
@inline get_centers(bo::BinnedObject, dim::Int=1) = collect(bo.bins[dim])

"""
    values(bo::BinnedObject)

Return the underlying array of bin values.
"""
@inline Base.values(bo::BinnedObject) = bo.values
@inline get_values(bo::BinnedObject) = Base.values(bo)

# access via lw() syntax
@inline function (lw::BinnedObject{1,T,B})(x::Real) where {T,B}
    idx = _binindex(lw.bins[1], x)
    lw.values[idx]
end

@inline logweight(bo::BinnedObject{1}, x::Real) = bo(x)
@inline logweight(bo::BinnedObject{N}, xs::Vararg{Real,N}) where {N} = bo(xs...)
@inline function (lw::BinnedObject{N,T,B})(xs::Vararg{Real,N}) where {N,T,B}
    idxs = _binindex(lw.bins, xs)
    lw.values[CartesianIndex(idxs)]
end

# access via lw[] syntax
@inline function Base.getindex(lw::BinnedObject{1,T,B}, x::Real) where {T,B}
    idx = _binindex(lw.bins[1], x)
    lw.values[idx]
end
@inline function Base.setindex!(lw::BinnedObject{1,T,B}, v, x::Real) where {T,B}
    idx = _binindex(lw.bins[1], x)
    lw.values[idx] = v
end
@inline function Base.getindex(lw::BinnedObject{N,T,B}, xs::Vararg{Real,N}) where {N,T,B}
    #TODO: this can be very likely be optimized
    idxs = [_binindex(lw.bins[i], xs[i]) for i in 1:N]
    lw.values[idxs...]
end
@inline function Base.setindex!(lw::BinnedObject{N,T,B}, v, xs::Vararg{Real,N}) where {N,T,B}
    #TODO: this can be very likely be optimized
    idxs = [_binindex(lw.bins[i], xs[i]) for i in 1:N]
    lw.values[idxs...] = v
end

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