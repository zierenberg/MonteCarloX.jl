"""
    AbstractBlumeCapel <: AbstractSpinSystem

Base type for Blume-Capel-like models.

Implementations must provide:
- `propose_changes(sys, i, s_new) -> (Δpair, Δspin, Δspin2)`
- `modify!(sys, i, s_new, Δpair, Δspin, Δspin2)`
- `_cached_energy(sys)`, `_cached_magnetization(sys)`, `_full_energy(sys)`
"""
abstract type AbstractBlumeCapel <: AbstractSpinSystem end

using Graphs
using SparseArrays: SparseMatrixCSC, sparse

const _BC_STATES = Int8[-1, 0, 1]

@inline energy(sys::AbstractBlumeCapel; full=false) = full ? _full_energy(sys) : _cached_energy(sys)
@inline magnetization(sys::AbstractBlumeCapel; full=false) = full ? sum(sys.spins) : _cached_magnetization(sys)

@inline function _propose_state(rng, s_old::Int8)
    u = rand(rng, Bool)
    if s_old == Int8(-1)
        return u ? Int8(0) : Int8(1)
    elseif s_old == Int8(0)
        return u ? Int8(-1) : Int8(1)
    else
        return u ? Int8(-1) : Int8(0)
    end
end

@inline function _sqdiff(s_new::Int8, s_old::Int8)
    return Int(s_new) * Int(s_new) - Int(s_old) * Int(s_old)
end

@inline function delta_energy(sys::AbstractBlumeCapel, i, s_new::Int8)
    Δpair, Δspin, Δspin2 = propose_changes(sys, i, s_new)
    return delta_energy(sys, Δpair, Δspin, Δspin2, i)
end

function spin_flip!(sys::AbstractBlumeCapel, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = _propose_state(alg.rng, sys.spins[i])
    Δpair, Δspin, Δspin2 = propose_changes(sys, i, s_new)
    ΔE = delta_energy(sys, Δpair, Δspin, Δspin2, i)
    E_old = energy(sys)
    E_new = E_old + ΔE
    accept!(alg, E_new, E_old) && modify!(sys, i, s_new, Δpair, Δspin, Δspin2)
    return nothing
end

function spin_flip!(sys::AbstractBlumeCapel, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = _propose_state(alg.rng, sys.spins[i])
    Δpair, Δspin, Δspin2 = propose_changes(sys, i, s_new)
    ΔE = delta_energy(sys, Δpair, Δspin, Δspin2, i)
    accept!(alg, ΔE) && modify!(sys, i, s_new, Δpair, Δspin, Δspin2)
    return nothing
end

function spin_flip!(sys::AbstractBlumeCapel, alg::AbstractHeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    coupling = local_coupling(sys, i)
    h_i = site_field(sys, i)

    e1 = -(-1) * coupling - h_i * (-1) + sys.D
    e2 = 0.0
    e3 = -(1) * coupling - h_i * (1) + sys.D

    w1 = exp(-alg.β * e1)
    w2 = exp(-alg.β * e2)
    w3 = exp(-alg.β * e3)
    z = w1 + w2 + w3

    r = rand(alg.rng) * z
    s_new = r < w1 ? Int8(-1) : (r < (w1 + w2) ? Int8(0) : Int8(1))

    if s_new != sys.spins[i]
        Δpair, Δspin, Δspin2 = propose_changes(sys, i, s_new)
        modify!(sys, i, s_new, Δpair, Δspin, Δspin2)
    end

    alg.steps += 1
    return nothing
end

@inline function _init_spins_blume_capel!(spins::Vector{Int8}, type::Symbol; rng=nothing)
    if type == :up
        spins .= 1
    elseif type == :down
        spins .= -1
    elseif type == :zero
        spins .= 0
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        @inbounds for i in eachindex(spins)
            spins[i] = rand(rng, _BC_STATES)
        end
    else
        error("Unknown initialization type: $type")
    end
    return nothing
end

function init!(sys::AbstractBlumeCapel, type::Symbol; rng=nothing)
    _init_spins_blume_capel!(sys.spins, type; rng=rng)
    _recompute_cached!(sys)
    return sys
end

# -----------------------------------------------------------------------------
# Graph Blume-Capel (global J)
# -----------------------------------------------------------------------------

abstract type BlumeCapelGraph <: AbstractBlumeCapel end

mutable struct BlumeCapelGraphCouplingNoField{TJ<:Real,TD<:Real} <: BlumeCapelGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    D::TD
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
end

mutable struct BlumeCapelGraphCouplingUniformField{TJ<:Real,TD<:Real,TH<:Real} <: BlumeCapelGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    D::TD
    h::TH
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
    sum_field_interactions::Float64
end

mutable struct BlumeCapelGraphCouplingVectorField{TJ<:Real,TD<:Real,TH<:Real} <: BlumeCapelGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    D::TD
    h::Vector{TH}
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
    sum_field_interactions::Float64
end

function BlumeCapelGraph(graph::SimpleGraph, J::Real, D::Real; h=0)
    n = nv(graph)
    spins = ones(Int8, n)
    nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:n]

    if h isa Real
        if iszero(h)
            sys = BlumeCapelGraphCouplingNoField{typeof(J),typeof(D)}(spins, graph, nbrs, J, D, 0.0, n, n)
            _recompute_cached!(sys)
            return sys
        else
            hscalar = float(h)
            sys = BlumeCapelGraphCouplingUniformField{typeof(J),typeof(D),typeof(hscalar)}(
                spins, graph, nbrs, J, D, hscalar, 0.0, n, n, 0.0
            )
            _recompute_cached!(sys)
            return sys
        end
    elseif h isa AbstractVector{<:Real}
        @assert length(h) == n "Field vector length must match number of spins"
        hvec = collect(h)
        sys = BlumeCapelGraphCouplingVectorField{typeof(J),typeof(D),eltype(hvec)}(
            spins, graph, nbrs, J, D, hvec, 0.0, n, n, 0.0
        )
        _recompute_cached!(sys)
        return sys
    else
        error("h must be Real or AbstractVector{<:Real}")
    end
end

@inline function _graph_pair_sum_blume_capel(sys)
    pair_unweighted = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_unweighted += local_pair_interactions(sys, i)
    end
    return float(sys.J) * (pair_unweighted / 2)
end

@inline _pair_sum(sys::BlumeCapelGraph) = _graph_pair_sum_blume_capel(sys)

@inline _field_sum(sys::BlumeCapelGraphCouplingNoField) = 0.0
@inline _field_sum(sys::BlumeCapelGraphCouplingUniformField) = sys.h * sum(sys.spins)
function _field_sum(sys::BlumeCapelGraphCouplingVectorField)
    field_sum = 0.0
    @inbounds for i in eachindex(sys.spins)
        field_sum += sys.h[i] * sys.spins[i]
    end
    return field_sum
end

function _recompute_cached!(sys::BlumeCapelGraphCouplingNoField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_spins2 = sum(s -> Int(s) * Int(s), sys.spins)
    return nothing
end

function _recompute_cached!(sys::Union{BlumeCapelGraphCouplingUniformField,BlumeCapelGraphCouplingVectorField})
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_spins2 = sum(s -> Int(s) * Int(s), sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

@inline _cached_magnetization(sys::BlumeCapelGraph) = sys.sum_spins
@inline _cached_energy(sys::BlumeCapelGraphCouplingNoField) = -sys.sum_pair_interactions + sys.D * sys.sum_spins2
@inline _cached_energy(sys::Union{BlumeCapelGraphCouplingUniformField,BlumeCapelGraphCouplingVectorField}) =
    -sys.sum_pair_interactions - sys.sum_field_interactions + sys.D * sys.sum_spins2

@inline _full_energy(sys::BlumeCapelGraph) = -_pair_sum(sys) - _field_sum(sys) + sys.D * sum(s -> Int(s) * Int(s), sys.spins)

@inline function local_coupling(sys::BlumeCapelGraph, i)
    return float(sys.J) * sum(sys.spins[j] for j in sys.nbrs[i])
end

@inline site_field(sys::BlumeCapelGraphCouplingNoField, i) = 0.0
@inline site_field(sys::BlumeCapelGraphCouplingUniformField, i) = sys.h
@inline site_field(sys::BlumeCapelGraphCouplingVectorField, i) = sys.h[i]

@inline function propose_changes(sys::BlumeCapelGraph, i, s_new::Int8)
    s_old = sys.spins[i]
    Δspin = Int(s_new - s_old)
    Δspin2 = _sqdiff(s_new, s_old)
    Δpair = Δspin * local_coupling(sys, i)
    return Δpair, Δspin, Δspin2
end

@inline delta_field_interactions(sys::BlumeCapelGraphCouplingNoField, Δspin, i) = 0.0
@inline delta_field_interactions(sys::BlumeCapelGraphCouplingUniformField, Δspin, i) = sys.h * Δspin
@inline delta_field_interactions(sys::BlumeCapelGraphCouplingVectorField, Δspin, i) = sys.h[i] * Δspin

@inline delta_energy(sys::BlumeCapelGraph, Δpair, Δspin, Δspin2, i) =
    -Δpair - delta_field_interactions(sys, Δspin, i) + sys.D * Δspin2

function modify!(sys::BlumeCapelGraph, i, s_new::Int8, Δpair, Δspin, Δspin2)
    sys.spins[i] = s_new
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_spins2 += Δspin2
    return nothing
end

function modify!(sys::Union{BlumeCapelGraphCouplingUniformField,BlumeCapelGraphCouplingVectorField},
                 i,
                 s_new::Int8,
                 Δpair,
                 Δspin,
                 Δspin2)
    sys.spins[i] = s_new
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_spins2 += Δspin2
    sys.sum_field_interactions += delta_field_interactions(sys, Δspin, i)
    return nothing
end

# -----------------------------------------------------------------------------
# Matrix Blume-Capel (local J_ij)
# -----------------------------------------------------------------------------

abstract type BlumeCapelMatrix <: AbstractBlumeCapel end

mutable struct BlumeCapelMatrixCouplingNoField{TJ<:Real,TD<:Real} <: BlumeCapelMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    D::TD
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
end

mutable struct BlumeCapelMatrixCouplingUniformField{TJ<:Real,TD<:Real,TH<:Real} <: BlumeCapelMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    D::TD
    h::TH
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
    sum_field_interactions::Float64
end

mutable struct BlumeCapelMatrixCouplingVectorField{TJ<:Real,TD<:Real,TH<:Real} <: BlumeCapelMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    D::TD
    h::Vector{TH}
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_spins2::Int
    sum_field_interactions::Float64
end

function _check_square_blume_capel(J::SparseMatrixCSC{TJ,Int}) where {TJ<:Real}
    n = size(J, 1)
    @assert size(J, 2) == n "Sparse J must be square"
    return nothing
end

function _check_symmetric_blume_capel(J::SparseMatrixCSC{TJ,Int}) where {TJ<:Real}
    n = size(J, 1)
    for col in 1:n
        for ptr in J.colptr[col]:(J.colptr[col + 1] - 1)
            row = J.rowval[ptr]
            if row != col
                @assert J[row, col] == J[col, row] "Sparse J must be symmetric"
            end
        end
    end
    return nothing
end

function BlumeCapelMatrix(J::SparseMatrixCSC{TJ,Int}, D::Real; h=0) where {TJ<:Real}
    _check_square_blume_capel(J)
    _check_symmetric_blume_capel(J)
    n = size(J, 1)
    spins = ones(Int8, n)

    if h isa Real
        if iszero(h)
            sys = BlumeCapelMatrixCouplingNoField{TJ,typeof(D)}(spins, J, D, 0.0, n, n)
            _recompute_cached!(sys)
            return sys
        else
            hscalar = float(h)
            sys = BlumeCapelMatrixCouplingUniformField{TJ,typeof(D),typeof(hscalar)}(spins, J, D, hscalar, 0.0, n, n, 0.0)
            _recompute_cached!(sys)
            return sys
        end
    elseif h isa AbstractVector{<:Real}
        @assert length(h) == n "Field vector length must match number of spins"
        hvec = collect(h)
        sys = BlumeCapelMatrixCouplingVectorField{TJ,typeof(D),eltype(hvec)}(spins, J, D, hvec, 0.0, n, n, 0.0)
        _recompute_cached!(sys)
        return sys
    else
        error("h must be Real or AbstractVector{<:Real}")
    end
end

@inline function local_pair_interactions(sys::BlumeCapelMatrix, i)
    s_i = sys.spins[i]
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i + 1] - 1)
        j = sys.J.rowval[ptr]
        if j != i
            acc += s_i * sys.J.nzval[ptr] * sys.spins[j]
        end
    end
    return acc
end

@inline function local_coupling(sys::BlumeCapelMatrix, i)
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i + 1] - 1)
        j = sys.J.rowval[ptr]
        if j != i
            acc += sys.J.nzval[ptr] * sys.spins[j]
        end
    end
    return acc
end

function _pair_sum(sys::BlumeCapelMatrix)
    pair_sum = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_sum += local_pair_interactions(sys, i)
    end
    return pair_sum / 2
end

@inline _field_sum(sys::BlumeCapelMatrixCouplingNoField) = 0.0
@inline _field_sum(sys::BlumeCapelMatrixCouplingUniformField) = sys.h * sum(sys.spins)
function _field_sum(sys::BlumeCapelMatrixCouplingVectorField)
    field_sum = 0.0
    @inbounds for i in eachindex(sys.spins)
        field_sum += sys.h[i] * sys.spins[i]
    end
    return field_sum
end

function _recompute_cached!(sys::BlumeCapelMatrixCouplingNoField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_spins2 = sum(s -> Int(s) * Int(s), sys.spins)
    return nothing
end

function _recompute_cached!(sys::Union{BlumeCapelMatrixCouplingUniformField,BlumeCapelMatrixCouplingVectorField})
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_spins2 = sum(s -> Int(s) * Int(s), sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

@inline _cached_magnetization(sys::BlumeCapelMatrix) = sys.sum_spins
@inline _cached_energy(sys::BlumeCapelMatrixCouplingNoField) = -sys.sum_pair_interactions + sys.D * sys.sum_spins2
@inline _cached_energy(sys::Union{BlumeCapelMatrixCouplingUniformField,BlumeCapelMatrixCouplingVectorField}) =
    -sys.sum_pair_interactions - sys.sum_field_interactions + sys.D * sys.sum_spins2

@inline _full_energy(sys::BlumeCapelMatrix) = -_pair_sum(sys) - _field_sum(sys) + sys.D * sum(s -> Int(s) * Int(s), sys.spins)

@inline site_field(sys::BlumeCapelMatrixCouplingNoField, i) = 0.0
@inline site_field(sys::BlumeCapelMatrixCouplingUniformField, i) = sys.h
@inline site_field(sys::BlumeCapelMatrixCouplingVectorField, i) = sys.h[i]

@inline function propose_changes(sys::BlumeCapelMatrix, i, s_new::Int8)
    s_old = sys.spins[i]
    Δspin = Int(s_new - s_old)
    Δspin2 = _sqdiff(s_new, s_old)
    Δpair = Δspin * local_coupling(sys, i)
    return Δpair, Δspin, Δspin2
end

@inline delta_field_interactions(sys::BlumeCapelMatrixCouplingNoField, Δspin, i) = 0.0
@inline delta_field_interactions(sys::BlumeCapelMatrixCouplingUniformField, Δspin, i) = sys.h * Δspin
@inline delta_field_interactions(sys::BlumeCapelMatrixCouplingVectorField, Δspin, i) = sys.h[i] * Δspin

@inline delta_energy(sys::BlumeCapelMatrix, Δpair, Δspin, Δspin2, i) =
    -Δpair - delta_field_interactions(sys, Δspin, i) + sys.D * Δspin2

function modify!(sys::BlumeCapelMatrix, i, s_new::Int8, Δpair, Δspin, Δspin2)
    sys.spins[i] = s_new
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_spins2 += Δspin2
    return nothing
end

function modify!(sys::Union{BlumeCapelMatrixCouplingUniformField,BlumeCapelMatrixCouplingVectorField},
                 i,
                 s_new::Int8,
                 Δpair,
                 Δspin,
                 Δspin2)
    sys.spins[i] = s_new
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_spins2 += Δspin2
    sys.sum_field_interactions += delta_field_interactions(sys, Δspin, i)
    return nothing
end

# Backward-compatible convenience method.
@inline function modify!(sys::AbstractBlumeCapel, i, s_new, _)
    Δpair, Δspin, Δspin2 = propose_changes(sys, i, Int8(s_new))
    return modify!(sys, i, Int8(s_new), Δpair, Δspin, Δspin2)
end

# -----------------------------------------------------------------------------
# General Blume-Capel constructor (factory)
# -----------------------------------------------------------------------------

function BlumeCapel(graph::SimpleGraph, J::Real, D::Real; h=0)
    return BlumeCapelGraph(graph, J, D; h=h)
end

function BlumeCapel(J::SparseMatrixCSC, D::Real; h=0)
    return BlumeCapelMatrix(J, D; h=h)
end

function BlumeCapel(graph::SimpleGraph, J::AbstractVector{<:Real}, D::Real; h=0)
    @assert ne(graph) == length(J) "Length of J vector must equal number of graph edges"

    TJ = float(eltype(J))
    rows = Vector{Int}(undef, 2ne(graph))
    cols = Vector{Int}(undef, 2ne(graph))
    vals = Vector{TJ}(undef, 2ne(graph))

    k = 1
    for (idx, e) in enumerate(edges(graph))
        i = src(e)
        j = dst(e)
        Jij = TJ(J[idx])
        rows[k] = i; cols[k] = j; vals[k] = Jij; k += 1
        rows[k] = j; cols[k] = i; vals[k] = Jij; k += 1
    end

    n = nv(graph)
    Jmat = sparse(rows, cols, vals, n, n)
    return BlumeCapelMatrix(Jmat, D; h=h)
end

function BlumeCapel(dims::Vector{Int}; J=1, D=0, periodic=true, h=0)
    graph = Graphs.SimpleGraphs.grid(dims; periodic)
    if J isa Real
        return BlumeCapelGraph(graph, J, D; h=h)
    elseif J isa AbstractVector{<:Real}
        return BlumeCapel(graph, J, D; h=h)
    elseif J isa SparseMatrixCSC
        return BlumeCapelMatrix(J, D; h=h)
    else
        error("J must be Real, AbstractVector{<:Real}, or SparseMatrixCSC")
    end
end
