"""
    TabulatedLogWeight <: AbstractLogWeight

Mutable tabulated log weight backed by a histogram.

Calling with a scalar state value returns the current tabulated log weight
at the corresponding histogram bin.
"""
mutable struct TabulatedLogWeight{T<:AbstractFloat,N,E} <: AbstractLogWeight
    table::Histogram{T,N,E}
end

@inline (lw::TabulatedLogWeight)(x::Real) = lw.table[x]
@inline Base.getindex(lw::TabulatedLogWeight, x::Real) = lw.table[x]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, x::Real) = (lw.table[x] = value)
@inline Base.getindex(lw::TabulatedLogWeight, idx...) = lw.table.weights[idx...]
@inline Base.setindex!(lw::TabulatedLogWeight, value::Real, idx...) = (lw.table.weights[idx...] = value)

@inline Base.size(lw::TabulatedLogWeight) = size(lw.table.weights)

function Base.:-(lw::TabulatedLogWeight, rhs::AbstractArray)
    @assert size(lw.table.weights) == size(rhs)
    new_hist = deepcopy(lw.table)
    new_hist.weights .-= rhs
    return TabulatedLogWeight(new_hist)
end

function Base.:-(lw::TabulatedLogWeight, rhs::Histogram)
    @assert size(lw.table.weights) == size(rhs.weights)
    new_hist = deepcopy(lw.table)
    new_hist.weights .-= rhs.weights
    return TabulatedLogWeight(new_hist)
end

function Base.:-(lw::TabulatedLogWeight, rhs::Real)
    new_hist = deepcopy(lw.table)
    new_hist.weights .-= rhs
    return TabulatedLogWeight(new_hist)
end
