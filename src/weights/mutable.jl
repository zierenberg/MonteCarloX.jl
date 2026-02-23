"""
    TabulatedLogWeight <: AbstractLogWeight

Mutable tabulated log weight backed by a histogram.

Calling with a scalar state value returns the current tabulated log weight
at the corresponding histogram bin.
"""
mutable struct TabulatedLogWeight{T<:AbstractFloat,N,E} <: AbstractLogWeight
    histogram::Histogram{T,N,E}
end

function TabulatedLogWeight(edges::AbstractVector, init::Real)
    nbin = length(edges) - 1
    nbin > 0 || throw(ArgumentError("`edges` must contain at least two points"))
    weights = fill(Float64(init), nbin)
    return TabulatedLogWeight(Histogram(collect(edges), weights))
end

@inline (lw::TabulatedLogWeight)(x::Real) = lw.histogram[x]
@inline Base.getindex(lw::TabulatedLogWeight, x::Real) = lw.histogram[x]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, x::Real) = (lw.histogram[x] = value)
@inline Base.getindex(lw::TabulatedLogWeight, idx...) = lw.histogram.weights[idx...]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, idx...) = (lw.histogram.weights[idx...] = value)

@inline Base.size(lw::TabulatedLogWeight) = size(lw.histogram.weights)

@inline function _assert_same_domain(lw1::TabulatedLogWeight, lw2::TabulatedLogWeight)
    if lw1.histogram.edges != lw2.histogram.edges
        throw(ArgumentError("domain (edges of histogram) must match exactly"))
    end
    return nothing
end

