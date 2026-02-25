# binned log weights for discrete and continuous variables 
# (designed for histogram-based methods like multicanonical sampling and Wang-Landau)

abstract type AbstractBin end
@inline function _binindex(bins::NTuple{N,AbstractBin}, xs::NTuple{N,Real}) where N
    ntuple(i -> _binindex(bins[i], xs[i]), N)
end

# TODO: final adaptation may be to make the Binning high-dimensional instead of the bins

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
    BinnedLogWeight(domain::AbstractRange, init)

Construct a binned log weight object for the given domain and initial value.
The type of `domain` determines the specific binned log weight type:

# Examples
```julia
# Discrete 1D
lw1d = BinnedLogWeight(0:10, 0.0)
# Discrete ND
lw2d = BinnedLogWeight((0:5, 0:5), 0.0)
# Continuous 1D
lw1d_cont = BinnedLogWeight(0.0:0.5:5.0, 0.0)
# Continuous ND
lw2d_cont = BinnedLogWeight((0.0:0.5:5.0, 0.0:0.5:5.0), 0.0)
``` 
"""
struct BinnedLogWeight{N,B<:AbstractBin} <: AbstractLogWeight
    weights :: Array{Float64,N}
    bins :: NTuple{N,B}
end

function BinnedLogWeight(domain::AbstractRange{T}, init::S) where {T<:Real, S}
    return BinnedLogWeight((domain,), init)
end

function BinnedLogWeight(domain::AbstractVector{T}, init::S) where {T<:Real, S}
    return BinnedLogWeight((domain,), init)
end

function BinnedLogWeight(domains::NTuple{N,Union{AbstractRange{T},AbstractVector{T}}}, init::Real) where {N,T<:Real}
    bins = ntuple(i -> begin
        d = domains[i]
        if eltype(d) <: Integer
            if d isa AbstractRange
                DiscreteBinning(first(d), T(step(d)), length(d))
            elseif d isa AbstractVector
                n = length(d)
                if n < 2
                    DiscreteBinning(d, zero(T), 1)
                else
                    steps = diff(d)
                    if all(steps .== steps[1])
                        DiscreteBinning(d, steps[1], n)
                    else
                        error("Non-equidistant discrete bins not supported without ExplicitBinning.")
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
    weights = fill(init, sizes...)
    return BinnedLogWeight{N,typeof(bins[1])}(weights, bins)
end
# catch for invalid domain types
function BinnedLogWeight(domain, init)
    throw(ArgumentError("Invalid domain type for BinnedLogWeight: Expected AbstractRange or AbstractVector of Real numbers, or a tuple of such (**but with identical types**)."))
end
# default constructor
@inline BinnedLogWeight(domain) = BinnedLogWeight(domain, 0.0)

# size of the weights array
@inline Base.size(lw::BinnedLogWeight) = size(lw.weights)

# access via lw() syntax
@inline function (lw::BinnedLogWeight{1,B})(x::Real) where B
    idx = _binindex(lw.bins[1], x)
    @inbounds lw.weights[idx]
end
@inline function (lw::BinnedLogWeight{N,B})(xs::Vararg{Real,N}) where {N,B}
    idxs = _binindex(lw.bins, xs)
    @inbounds lw.weights[CartesianIndex(idxs)]
end

# access via lw[] syntax
@inline function Base.getindex(lw::BinnedLogWeight{1,B}, x::Real) where B
    idx = _binindex(lw.bins[1], x)
    @inbounds lw.weights[idx]
end
@inline function Base.setindex!(lw::BinnedLogWeight{1,B}, v, x::Real) where B
    idx = _binindex(lw.bins[1], x)
    @inbounds lw.weights[idx] = v
end
@inline function Base.getindex(lw::BinnedLogWeight{N,B}, xs::Vararg{Real,N}) where {N,B}
    #TODO: this can be very likely be optimized
    idxs = [_binindex(lw.bins[i], xs[i]) for i in 1:N]
    @inbounds lw.weights[idxs...]
end
@inline function Base.setindex!(lw::BinnedLogWeight{N,B}, v, xs::Vararg{Real,N}) where {N,B}
    #TODO: this can be very likely be optimized
    idxs = [_binindex(lw.bins[i], xs[i]) for i in 1:N]
    @inbounds lw.weights[idxs...] = v
end

"""
    zero(lw::BinnedLogWeight)

Return a new `BinnedLogWeight` of the same bins as `lw` but with all weights set to zero.
"""
function Base.zero(lw::BinnedLogWeight)
    new_weights = fill(0.0, size(lw.weights))
    return BinnedLogWeight{length(lw.bins),typeof(lw.bins[1])}(new_weights, lw.bins)
end

# helper to check if two BinnedLogWeight objects have the same binning structure
@inline function _assert_same_domain(lw1::BinnedLogWeight, lw2::BinnedLogWeight)
    @assert length(lw1.bins) == length(lw2.bins) "BinnedLogWeight objects must have the same number of dimensions."
    for i in 1:length(lw1.bins)
        b1, b2 = lw1.bins[i], lw2.bins[i]
        @assert typeof(b1) == typeof(b2) "Bin types must match in each dimension."
        if b1 isa DiscreteBinning
            @assert b1.start == b2.start "Discrete bins must have the same start."
            @assert b1.step == b2.step "Discrete bins must have the same step."
            @assert b1.num == b2.num "Discrete bins must have the same num."
        elseif b1 isa ContinuousBinning
            @assert b1.edges == b2.edges "Continuous bins must have the same edges."
        else
            error("Unknown bin type: $(typeof(b1))")
        end
    end
    return nothing
end