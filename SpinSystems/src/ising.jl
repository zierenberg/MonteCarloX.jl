"""
    AbstractIsing <: AbstractSpinSystem

Base type for Ising-like models.

Implementations must provide:
- `flip_changes(sys, i) -> (Δpair, Δspin)`
- `modify!(sys, i, Δpair, Δspin)`
- `_cached_energy(sys)`, `_cached_magnetization(sys)`, `_full_energy(sys)`
"""
abstract type AbstractIsing <: AbstractSpinSystem end

using Graphs
using SparseArrays: SparseMatrixCSC, sparse

# Shared observables and update
@inline energy(sys::AbstractIsing; full=false) = full ? _full_energy(sys) : _cached_energy(sys)
@inline magnetization(sys::AbstractIsing; full=false) = full ? sum(sys.spins) : _cached_magnetization(sys)

@inline function delta_energy(sys::AbstractIsing, i)
    Δpair, Δspin = flip_changes(sys, i)
    return delta_energy(sys, Δpair, Δspin, i)
end

function spin_flip!(sys::AbstractIsing, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    Δpair, Δspin = flip_changes(sys, i)
    ΔE = delta_energy(sys, Δpair, Δspin, i)
    E_old = energy(sys)
    E_new = E_old + ΔE
    log_ratio = log_acceptance_ratio(alg, E_new, E_old)
    accept!(alg, log_ratio) && modify!(sys, i, Δpair, Δspin)
    return nothing
end

function spin_flip!(sys::AbstractIsing, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    Δpair, Δspin = flip_changes(sys, i)
    ΔE = delta_energy(sys, Δpair, Δspin, i)
    log_ratio = log_acceptance_ratio(alg, ΔE)
    accept!(alg, log_ratio) && modify!(sys, i, Δpair, Δspin)
    return nothing
end

function spin_flip!(sys::AbstractIsing, alg::AbstractHeatBath)
    i = pick_site(alg.rng, length(sys.spins))
    s_old = sys.spins[i]
    Δpair, Δspin = flip_changes(sys, i)
    ΔE = delta_energy(sys, Δpair, Δspin, i)

    p_plus = logistic(alg.β * float(s_old) * ΔE)
    s_new = rand(alg.rng) < p_plus ? Int8(1) : Int8(-1)

    if s_new != s_old
        modify!(sys, i, Δpair, Δspin)
    end

    alg.steps += 1
    return nothing
end

@inline function _init_spins!(spins::Vector{Int8}, type::Symbol; rng=nothing)
    if type == :up
        spins .= 1
    elseif type == :down
        spins .= -1
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        spins .= rand(rng, [-1, 1], length(spins))
    else
        error("Unknown initialization type: $type")
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Graph Ising (global J)
# -----------------------------------------------------------------------------

abstract type IsingGraph <: AbstractIsing end

mutable struct IsingGraphCouplingNoField{TJ<:Real} <: IsingGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    sum_pair_interactions::TJ
    sum_spins::Int
end

mutable struct IsingGraphCouplingUniformField{TJ<:Real,TH<:Real} <: IsingGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    h::TH
    sum_pair_interactions::TJ
    sum_spins::Int
    sum_field_interactions::Float64
end

mutable struct IsingGraphCouplingVectorField{TJ<:Real,TH<:Real} <: IsingGraph
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::TJ
    h::Vector{TH}
    sum_pair_interactions::TJ
    sum_spins::Int
    sum_field_interactions::Float64
end

function IsingGraph(graph::SimpleGraph, J::Real; h=0)
    n = nv(graph)
    spins = ones(Int8, n)
    nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:n]
    pair0 = J * ne(graph)

    if h isa Real
        if iszero(h)
            return IsingGraphCouplingNoField{typeof(J)}(spins, graph, nbrs, J, pair0, n)
        else
            hscalar = float(h)
            return IsingGraphCouplingUniformField{typeof(J),typeof(hscalar)}(
                spins, graph, nbrs, J, hscalar, pair0, n, hscalar * n
            )
        end
    elseif h isa AbstractVector{<:Real}
        @assert length(h) == n "Field vector length must match number of spins"
        hvec = collect(h)
        return IsingGraphCouplingVectorField{typeof(J),eltype(hvec)}(spins, graph, nbrs, J, hvec, pair0, n, sum(hvec))
    else
        error("h must be Real or AbstractVector{<:Real}")
    end
end

function init!(sys::IsingGraph, type::Symbol; rng=nothing)
    _init_spins!(sys.spins, type; rng=rng)
    _recompute_cached!(sys)
    return sys
end

function _recompute_cached!(sys::IsingGraphCouplingNoField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    return nothing
end

function _recompute_cached!(sys::IsingGraphCouplingUniformField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

function _recompute_cached!(sys::IsingGraphCouplingVectorField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

@inline function _graph_pair_sum(sys)
    pair_unweighted = 0
    @inbounds for i in eachindex(sys.spins)
        pair_unweighted += local_pair_interactions(sys, i)
    end
    return sys.J isa Integer ? (sys.J * div(pair_unweighted, 2)) : typeof(sys.J)(sys.J * (pair_unweighted / 2))
end

@inline _pair_sum(sys::IsingGraph) = _graph_pair_sum(sys)
@inline _field_sum(sys::IsingGraphCouplingNoField) = 0.0
@inline _field_sum(sys::IsingGraphCouplingUniformField) = sys.h * sum(sys.spins)
function _field_sum(sys::IsingGraphCouplingVectorField)
    field_sum = 0.0
    @inbounds for i in eachindex(sys.spins)
        field_sum += sys.h[i] * sys.spins[i]
    end
    return field_sum
end

@inline _cached_magnetization(sys::IsingGraph) = sys.sum_spins
@inline _cached_energy(sys::IsingGraphCouplingNoField) = -sys.sum_pair_interactions
@inline _cached_energy(sys::IsingGraphCouplingUniformField) = -sys.sum_pair_interactions - sys.sum_field_interactions
@inline _cached_energy(sys::IsingGraphCouplingVectorField) = -sys.sum_pair_interactions - sys.sum_field_interactions

@inline _full_energy(sys::IsingGraph) = -_pair_sum(sys) - _field_sum(sys)

@inline function flip_changes(sys::IsingGraph, i)
    s = sys.spins[i]
    Δpair = -2 * sys.J * local_pair_interactions(sys, i)
    Δspin = -2 * s
    return Δpair, Δspin
end

@inline delta_field_interactions(sys::IsingGraphCouplingNoField, Δspin, i) = 0.0
@inline delta_field_interactions(sys::IsingGraphCouplingUniformField, Δspin, i) = sys.h * Δspin
@inline delta_field_interactions(sys::IsingGraphCouplingVectorField, Δspin, i) = sys.h[i] * Δspin

@inline delta_energy(sys::IsingGraph, Δpair, Δspin, i) = -Δpair - delta_field_interactions(sys, Δspin, i)

function modify!(sys::IsingGraph, i, Δpair, Δspin)
    sys.spins[i] = -sys.spins[i]
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    return nothing
end

function modify!(sys::Union{IsingGraphCouplingUniformField,IsingGraphCouplingVectorField}, i, Δpair, Δspin)
    sys.spins[i] = -sys.spins[i]
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_field_interactions += delta_field_interactions(sys, Δspin, i)
    return nothing
end

# -----------------------------------------------------------------------------
# Matrix Ising (local J_{ij})
# -----------------------------------------------------------------------------

abstract type IsingMatrix <: AbstractIsing end

mutable struct IsingMatrixCouplingNoField{TJ<:Real} <: IsingMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    sum_pair_interactions::Float64
    sum_spins::Int
end

mutable struct IsingMatrixCouplingUniformField{TJ<:Real,TH<:Real} <: IsingMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    h::TH
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_field_interactions::Float64
end

mutable struct IsingMatrixCouplingVectorField{TJ<:Real,TH<:Real} <: IsingMatrix
    spins::Vector{Int8}
    J::SparseMatrixCSC{TJ,Int}
    h::Vector{TH}
    sum_pair_interactions::Float64
    sum_spins::Int
    sum_field_interactions::Float64
end

function _check_square(J::SparseMatrixCSC{TJ,Int}) where {TJ<:Real}
    n = size(J, 1)
    @assert size(J, 2) == n "Sparse J must be square"
    return nothing
end

function _check_symmetric(J::SparseMatrixCSC{TJ,Int}) where {TJ<:Real}
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

function IsingMatrix(J::SparseMatrixCSC{TJ,Int}; h=0) where {TJ<:Real}
    _check_square(J)
    _check_symmetric(J)
    n = size(J, 1)

    spins = ones(Int8, n)

    if h isa Real
        if iszero(h)
            sys = IsingMatrixCouplingNoField{TJ}(spins, J, 0.0, n)
            _recompute_cached!(sys)
            return sys
        else
            hscalar = h
            sys = IsingMatrixCouplingUniformField{TJ,typeof(hscalar)}(spins, J, hscalar, 0.0, n, 0.0)
            _recompute_cached!(sys)
            return sys
        end
    elseif h isa AbstractVector{<:Real}
        @assert length(h) == n "Field vector length must match number of spins"
        hvec = collect(h)
        sys = IsingMatrixCouplingVectorField{TJ,eltype(hvec)}(spins, J, hvec, 0.0, n, 0.0)
        _recompute_cached!(sys)
        return sys
    else
        error("h must be Real or AbstractVector{<:Real}")
    end
end

function init!(sys::IsingMatrix, type::Symbol; rng=nothing)
    _init_spins!(sys.spins, type; rng=rng)
    _recompute_cached!(sys)
    return sys
end

@inline function local_pair_interactions(sys::IsingMatrix, i)
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

function _pair_sum(sys::IsingMatrix)
    pair_sum = 0.0
    for i in eachindex(sys.spins)
        pair_sum += local_pair_interactions(sys, i)
    end
    return pair_sum / 2
end

@inline _field_sum(sys::IsingMatrixCouplingNoField) = 0.0
@inline _field_sum(sys::IsingMatrixCouplingUniformField) = sys.h * sum(sys.spins)
function _field_sum(sys::IsingMatrixCouplingVectorField)
    field_sum = 0.0
    @inbounds for i in eachindex(sys.spins)
        field_sum += sys.h[i] * sys.spins[i]
    end
    return field_sum
end

function _recompute_cached!(sys::IsingMatrixCouplingNoField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    return nothing
end

function _recompute_cached!(sys::IsingMatrixCouplingUniformField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

function _recompute_cached!(sys::IsingMatrixCouplingVectorField)
    sys.sum_pair_interactions = _pair_sum(sys)
    sys.sum_spins = sum(sys.spins)
    sys.sum_field_interactions = _field_sum(sys)
    return nothing
end

@inline _cached_magnetization(sys::IsingMatrix) = sys.sum_spins
@inline _cached_energy(sys::IsingMatrixCouplingNoField) = -sys.sum_pair_interactions
@inline _cached_energy(sys::IsingMatrixCouplingUniformField) = -sys.sum_pair_interactions - sys.sum_field_interactions
@inline _cached_energy(sys::IsingMatrixCouplingVectorField) = -sys.sum_pair_interactions - sys.sum_field_interactions

@inline _full_energy(sys::IsingMatrix) = -_pair_sum(sys) - _field_sum(sys)

@inline function flip_changes(sys::IsingMatrix, i)
    s = sys.spins[i]
    Δpair = -2.0 * local_pair_interactions(sys, i)
    Δspin = -2 * s
    return Δpair, Δspin
end

@inline delta_field_interactions(sys::IsingMatrixCouplingNoField, Δspin, i) = 0.0
@inline delta_field_interactions(sys::IsingMatrixCouplingUniformField, Δspin, i) = sys.h * Δspin
@inline delta_field_interactions(sys::IsingMatrixCouplingVectorField, Δspin, i) = sys.h[i] * Δspin

@inline delta_energy(sys::IsingMatrix, Δpair, Δspin, i) = -Δpair - delta_field_interactions(sys, Δspin, i)

function modify!(sys::IsingMatrix, i, Δpair, Δspin)
    sys.spins[i] = -sys.spins[i]
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    return nothing
end

function modify!(sys::Union{IsingMatrixCouplingUniformField,IsingMatrixCouplingVectorField}, i, Δpair, Δspin)
    sys.spins[i] = -sys.spins[i]
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    sys.sum_field_interactions += delta_field_interactions(sys, Δspin, i)
    return nothing
end

# -----------------------------------------------------------------------------
# General Ising constructor (factory)
# -----------------------------------------------------------------------------

function Ising(graph::SimpleGraph, J::Real; h=0)
    return IsingGraph(graph, J; h=h)
end

function Ising(J::SparseMatrixCSC; h=0)
    return IsingMatrix(J; h=h)
end

function Ising(graph::SimpleGraph, J::AbstractVector{<:Real}; h=0)
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
    Jmat =sparse(rows, cols, vals, n, n) 
    return IsingMatrix(Jmat; h=h)
end

function Ising(dims::Vector{Int}; J=1, periodic=true, h=0)
    graph = Graphs.SimpleGraphs.grid(dims; periodic)
    if J isa Real
        return IsingGraph(graph, J; h=h)
    elseif J isa AbstractVector{<:Real}
        return Ising(graph, J; h=h)
    elseif J isa SparseMatrixCSC
        return IsingMatrix(J; h=h)
    else
        error("J must be Real, AbstractVector{<:Real}, or SparseMatrixCSC")
    end
end

# -----------------------------------------------------------------------------
# Optimized 2D Ising Model (J=1, periodic, no field)
# -----------------------------------------------------------------------------

mutable struct IsingLatticeOptim <: AbstractIsing
    spins::Vector{Int8}
    const Lx::Int
    const Ly::Int
    const N::Int
    const nbr4::Vector{NTuple{4,Int}}
    sum_pair_interactions::Int
    sum_spins::Int
end

function IsingLatticeOptim(Lx::Int, Ly::Int)
    N = Lx * Ly
    nbr4 = Vector{NTuple{4,Int}}(undef, N)

    @inbounds for y in 1:Ly
        y_up = (y == 1)  ? Ly : (y - 1)
        y_dn = (y == Ly) ? 1  : (y + 1)
        row = (y - 1) * Lx
        row_up = (y_up - 1) * Lx
        row_dn = (y_dn - 1) * Lx
        for x in 1:Lx
            i = row + x
            x_l = (x == 1)  ? Lx : (x - 1)
            x_r = (x == Lx) ? 1  : (x + 1)
            nbr4[i] = (row + x_l, row + x_r, row_up + x, row_dn + x)
        end
    end

    return IsingLatticeOptim(ones(Int8, N), Lx, Ly, N, nbr4, 2N, N)
end

@inline function local_pair_interactions(sys::IsingLatticeOptim, i)
    @inbounds begin
        s = sys.spins[i]
        n1, n2, n3, n4 = sys.nbr4[i]
        return s * (sys.spins[n1] + sys.spins[n2] + sys.spins[n3] + sys.spins[n4])
    end
end

@inline _cached_magnetization(sys::IsingLatticeOptim) = sys.sum_spins
@inline _cached_energy(sys::IsingLatticeOptim) = -sys.sum_pair_interactions

function _full_energy(sys::IsingLatticeOptim)
    s = 0
    for i in eachindex(sys.spins)
        s += local_pair_interactions(sys, i)
    end
    return -div(s, 2)
end

@inline function flip_changes(sys::IsingLatticeOptim, i)
    s = sys.spins[i]
    Δpair = -2 * local_pair_interactions(sys, i)
    Δspin = -2 * s
    return Δpair, Δspin
end

@inline delta_energy(sys::IsingLatticeOptim, Δpair, Δspin, i) = -Δpair

function modify!(sys::IsingLatticeOptim, i, Δpair, Δspin)
    sys.spins[i] = -sys.spins[i]
    sys.sum_pair_interactions += Δpair
    sys.sum_spins += Δspin
    return nothing
end
