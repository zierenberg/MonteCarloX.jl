export BinnedLogWeight

abstract type AbstractBinnedLogWeight <: AbstractLogWeight end

############################
# Discrete bins (1D)
############################

"""
    BinnedLogWeight(domain::AbstractRange{<:Integer}, init)

Discrete binned log weight with arbitrary step size.

Examples:
- `-128:4:128`
- `0:2:100`

Optimized for O(1) lookup.
"""
mutable struct BinnedLogWeight{T} <: AbstractBinnedLogWeight
    weights :: Vector{T}
    start   :: Int
    step    :: Int
end

@inline function BinnedLogWeight(domain::AbstractRange{<:Integer},
                                 init::T) where T
    _step = Int(step(domain))
    BinnedLogWeight(fill(init, length(domain)),
                    first(domain),
                    _step)
end

@inline function binindex(lw::BinnedLogWeight, x::Integer)
    @inbounds Int((x - lw.start) ÷ lw.step) + 1
end

@inline (lw::BinnedLogWeight)(x::Integer) =
    @inbounds lw.weights[binindex(lw, x)]

@inline function Base.setindex!(lw::BinnedLogWeight, v, x::Integer)
    @inbounds lw.weights[binindex(lw, x)] = v
end

############################
# Discrete bins (ND)
############################

"""
    BinnedLogWeight(domains::NTuple{N,AbstractRange{<:Integer}}, init)

N-dimensional discrete binned log weight with arbitrary steps.
"""
mutable struct BinnedLogWeightND{T,N} <: AbstractBinnedLogWeight
    weights :: Array{T,N}
    start   :: NTuple{N,Int}
    step    :: NTuple{N,Int}
end

function BinnedLogWeight(domains::NTuple{N,AbstractRange{<:Integer}},
                         init::T) where {N,T}
    sizes = map(length, domains)
    start = ntuple(i -> first(domains[i]), N)
    _step  = ntuple(i -> Int(step(domains[i])), N)
    weights = fill(init, sizes...)
    BinnedLogWeightND(weights, start, _step)
end

@inline function binindex(lw::BinnedLogWeightND{T,N},
                          xs::NTuple{N,Integer}) where {T,N}
    CartesianIndex(ntuple(i ->
        Int((xs[i] - lw.start[i]) ÷ lw.step[i]) + 1, N))
end

@inline (lw::BinnedLogWeightND{T,N})(xs::Vararg{Integer,N}) where {T,N} =
    @inbounds lw.weights[binindex(lw, xs)]

############################
# Continuous bins (1D)
############################

"""
    BinnedLogWeight(edges::AbstractVector{<:Real}, init)

One-dimensional continuous binned log weight.
Histogram-style semantics.
"""
mutable struct BinnedLogWeight1D{T} <: AbstractBinnedLogWeight
    weights :: Vector{T}
    edges   :: Vector{T}
    centers :: Vector{T}
end

function BinnedLogWeight(edges::AbstractVector{T},
                         init::T) where T<:Real
    nbins = length(edges) - 1
    weights = fill(init, nbins)
    centers = @inbounds (edges[1:end-1] .+ edges[2:end]) .* T(0.5)
    BinnedLogWeight1D(weights, collect(edges), centers)
end

@inline function (lw::BinnedLogWeight1D)(x::Real)
    i = searchsortedlast(lw.edges, x)
    @inbounds lw.weights[clamp(i, 1, length(lw.weights))]
end

@inline function Base.setindex!(lw::BinnedLogWeight1D, v, x::Real)
    i = searchsortedlast(lw.edges, x)
    @inbounds lw.weights[clamp(i, 1, length(lw.weights))] = v
end

############################
# Continuous bins (ND)
############################

"""
    BinnedLogWeight(edges::NTuple{N,AbstractVector{<:Real}}, init)

N-dimensional continuous binned log weight.
"""
mutable struct BinnedLogWeightContinuousND{T,N} <: AbstractBinnedLogWeight
    weights :: Array{T,N}
    edges   :: NTuple{N,Vector{T}}
    centers :: NTuple{N,Vector{T}}
end

function BinnedLogWeight(edges::NTuple{N,AbstractVector{T}},
                         init::T) where {N,T<:Real}
    sizes   = ntuple(i -> length(edges[i]) - 1, N)
    weights = fill(init, sizes)
    centers = ntuple(i ->
        @inbounds (edges[i][1:end-1] .+ edges[i][2:end]) .* T(0.5), N)
    BinnedLogWeightContinuousND(weights, map(collect, edges), centers)
end

@inline function (lw::BinnedLogWeightContinuousND{T,N})(
    xs::Vararg{Real,N}) where {T,N}

    I = CartesianIndex(ntuple(i ->
        clamp(searchsortedlast(lw.edges[i], xs[i]),
              1, size(lw.weights, i)), N))
    @inbounds lw.weights[I]
end

############################
# Functional assignment
############################

"""
    lw[xs] = f

Assign weights by evaluating `f` at:
- discrete bins: `f(x)`
- continuous bins: `f(bin center)`
"""

@inline function Base.setindex!(lw::BinnedLogWeight,
                               f::Function, xs)
    @inbounds for x in xs
        lw[x] = f(x)
    end
    lw
end

@inline function Base.setindex!(lw::BinnedLogWeightND,
                               f::Function, xs)
    for x in xs
        I = binindex(lw, x)
        @inbounds lw.weights[I] = f(x...)
    end
    lw
end

@inline function Base.setindex!(lw::BinnedLogWeight1D,
                               f::Function, xs)
    for x in xs
        i = clamp(searchsortedlast(lw.edges, x),
                  1, length(lw.weights))
        @inbounds lw.weights[i] = f(lw.centers[i])
    end
    lw
end

@inline function Base.setindex!(lw::BinnedLogWeightContinuousND{T,N},
                               f::Function, xs) where {T,N}
    for x in xs
        I = CartesianIndex(ntuple(i ->
            clamp(searchsortedlast(lw.edges[i], x[i]),
                  1, size(lw.weights, i)), N))
        @inbounds lw.weights[I] =
            f(ntuple(i -> lw.centers[i][I[i]], N)...)
    end
    lw
end

# 1D continuous indexing via []
@inline function Base.getindex(lw::BinnedLogWeight1D, x::Real)
    i = clamp(searchsortedlast(lw.edges, x), 1, length(lw.weights))
    @inbounds lw.weights[i]
end

@inline function Base.setindex!(lw::BinnedLogWeight1D, v::T, x::Real) where T
    i = clamp(searchsortedlast(lw.edges, x), 1, length(lw.weights))
    @inbounds lw.weights[i] = v
end

@inline function Base.getindex(lw::BinnedLogWeightContinuousND{T,N}, xs::Vararg{Real,N}) where {T,N}
    I = CartesianIndex(ntuple(i ->
        clamp(searchsortedlast(lw.edges[i], xs[i]),
              1, size(lw.weights, i)), N))
    @inbounds lw.weights[I]
end

@inline function Base.setindex!(lw::BinnedLogWeightContinuousND{T,N}, v::T, xs::Vararg{Real,N}) where {T,N}
    I = CartesianIndex(ntuple(i ->
        clamp(searchsortedlast(lw.edges[i], xs[i]),
              1, size(lw.weights, i)), N))
    @inbounds lw.weights[I] = v
end