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

@inline function _new_logweight_like(lw::TabulatedLogWeight, weights::AbstractArray)
    h = lw.histogram
    return TabulatedLogWeight(Histogram(h.edges, weights, h.closed, h.isdensity))
end

@inline Base.getproperty(lw::TabulatedLogWeight, name::Symbol) =
    name === :table ? getfield(lw, :histogram) : getfield(lw, name)

@inline Base.setproperty!(lw::TabulatedLogWeight, name::Symbol, value) =
    name === :table ? setfield!(lw, :histogram, value) : setfield!(lw, name, value)

Base.propertynames(::TabulatedLogWeight, private::Bool=false) =
    private ? (:histogram, :table) : (:histogram, :table)

@inline (lw::TabulatedLogWeight)(x::Real) = lw.histogram[x]
@inline Base.getindex(lw::TabulatedLogWeight, x::Real) = lw.histogram[x]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, x::Real) = (lw.histogram[x] = value)
@inline Base.getindex(lw::TabulatedLogWeight, idx...) = lw.histogram.weights[idx...]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, idx...) = (lw.histogram.weights[idx...] = value)

@inline Base.size(lw::TabulatedLogWeight) = size(lw.histogram.weights)

@inline function _assert_same_shape(weights::AbstractArray, rhs::AbstractArray)
    if size(weights) != size(rhs)
        throw(ArgumentError("weight-table shape mismatch: $(size(weights)) != $(size(rhs))"))
    end
    return nothing
end

@inline function _assert_same_histogram_bins(lhs::Histogram, rhs::Histogram)
    if lhs.edges != rhs.edges
        throw(ArgumentError("histogram bin edges must match exactly"))
    end
    return nothing
end

