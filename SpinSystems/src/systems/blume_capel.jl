abstract type AbstractBlumeCapel <: AbstractSpinSystem end

const _BC_STATES = Int8[-1, 0, 1]

# ── Shared observables ───────────────────────────────────────────────────────

@inline energy(sys::AbstractBlumeCapel; full=false) = full ? _full_energy(sys) : _cached_energy(sys)
@inline magnetization(sys::AbstractBlumeCapel; full=false) = full ? sum(sys.spins) : sys.cached_mag

# ── Shared initialization ────────────────────────────────────────────────────

function init!(sys::AbstractBlumeCapel, type::Symbol; rng=nothing)
    if type == :up
        sys.spins .= 1
    elseif type == :down
        sys.spins .= -1
    elseif type == :zero
        sys.spins .= 0
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        @inbounds for i in eachindex(sys.spins)
            sys.spins[i] = rand(rng, _BC_STATES)
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_cached!(sys)
    return sys
end

# ── Interface: propose_state ─────────────────────────────────────────────────

@inline function propose_state(rng, sys::AbstractBlumeCapel, i)
    @inbounds s_old = sys.spins[i]
    u = rand(rng, Bool)
    if s_old == Int8(-1)
        return u ? Int8(0) : Int8(1)
    elseif s_old == Int8(0)
        return u ? Int8(-1) : Int8(1)
    else
        return u ? Int8(-1) : Int8(0)
    end
end

# ── Shared delta_energy (uses _local_coupling dispatched per backend) ────────

@inline function delta_energy(sys::AbstractBlumeCapel, i, s_new::Int8)
    @inbounds s_old = Int(sys.spins[i])
    Δspin = Int(s_new) - s_old
    Δsq = Int(s_new)^2 - s_old^2
    coupling = _local_coupling(sys, i)
    return -Δspin * coupling - _site_field(sys.h, i) * Δspin + sys.crystal * Δsq
end

# ── Shared modify! ───────────────────────────────────────────────────────────

function MonteCarloX.modify!(sys::AbstractBlumeCapel, i::Int, s_new::Int8)
    @inbounds s_old = Int(sys.spins[i])
    Δspin = Int(s_new) - s_old
    Δsq = Int(s_new)^2 - s_old^2
    coupling = _local_coupling(sys, i)
    sys.spins[i] = s_new
    sys.cached_pair += Δspin * coupling
    sys.cached_mag += Δspin
    sys.cached_sq += Δsq
    _update_field_cache!(sys, sys.h, i, Float64(Δspin))
    return nothing
end

# ── Shared energy ────────────────────────────────────────────────────────────

@inline _cached_energy(sys::AbstractBlumeCapel) =
    -sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field) + float(sys.crystal) * sys.cached_sq

function _full_energy(sys::AbstractBlumeCapel)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return -(pair_full / 2) - _field_sum(sys.h, sys.spins) +
           float(sys.crystal) * sum(s -> Int(s)^2, sys.spins)
end

function _recompute_cached!(sys::AbstractBlumeCapel)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full / 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_sq = sum(s -> Int(s)^2, sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelLattice: D-dimensional periodic hypercubic lattice, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelLattice{D, NN, TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const nbrs::Vector{NTuple{NN,Int}}
    const J::TJ
    const crystal::TC
    const h::TH
    cached_pair::Float64    # J-weighted half sum
    cached_mag::Int
    cached_sq::Int          # Σ s_i²
    cached_field::Float64
end

function BlumeCapelLattice(dims::AbstractVector{<:Integer}, J::Real, crystal::Real; h=0)
    N = prod(dims)
    if h isa AbstractVector
        @assert length(h) == N "Field vector length must match number of sites"
        _bc_lattice(Tuple(Int.(dims)), J, crystal, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_lattice(Tuple(Int.(dims)), J, crystal, float(h))
    else
        _bc_lattice(Tuple(Int.(dims)), J, crystal, NoField())
    end
end

function _bc_lattice(dims::NTuple{D,Int}, J, crystal, h) where D
    N = prod(dims)
    nbrs = _build_lattice_neighbors(dims)
    sys = BlumeCapelLattice{D, 2D, typeof(J), typeof(crystal), typeof(h)}(
        ones(Int8, N), nbrs, J, crystal, h, 0.0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function _local_coupling(sys::BlumeCapelLattice{D}, i) where D
    @inbounds begin
        acc = 0
        for j in sys.nbrs[i]
            acc += sys.spins[j]
        end
        return float(sys.J) * acc
    end
end

@inline function local_pair_interactions(sys::BlumeCapelLattice{D}, i) where D
    @inbounds begin
        s = sys.spins[i]
        acc = 0
        for j in sys.nbrs[i]
            acc += sys.spins[j]
        end
        return float(sys.J) * s * acc
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelGraph: arbitrary graph, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelGraph{TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const graph::SimpleGraph
    const nbrs::Vector{Vector{Int}}
    const J::TJ
    const crystal::TC
    const h::TH
    cached_pair::Float64
    cached_mag::Int
    cached_sq::Int
    cached_field::Float64
end

function BlumeCapelGraph(graph::SimpleGraph, J::Real, crystal::Real; h=0)
    n = nv(graph)
    nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:n]
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _bc_graph(graph, nbrs, J, crystal, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_graph(graph, nbrs, J, crystal, float(h))
    else
        _bc_graph(graph, nbrs, J, crystal, NoField())
    end
end

function _bc_graph(graph, nbrs, J, crystal, h)
    n = nv(graph)
    sys = BlumeCapelGraph{typeof(J), typeof(crystal), typeof(h)}(
        ones(Int8, n), graph, nbrs, J, crystal, h, 0.0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function _local_coupling(sys::BlumeCapelGraph, i)
    acc = 0
    @inbounds for j in sys.nbrs[i]
        acc += sys.spins[j]
    end
    return float(sys.J) * acc
end

@inline function local_pair_interactions(sys::BlumeCapelGraph, i)
    s = sys.spins[i]
    acc = 0
    @inbounds for j in sys.nbrs[i]
        acc += sys.spins[j]
    end
    return float(sys.J) * s * acc
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelMatrix: sparse J_{ij}, arbitrary topology
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelMatrix{TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const J::SparseMatrixCSC{TJ,Int}
    const crystal::TC
    const h::TH
    cached_pair::Float64
    cached_mag::Int
    cached_sq::Int
    cached_field::Float64
end

function BlumeCapelMatrix(J::SparseMatrixCSC{TJ,Int}, crystal::Real; h=0) where {TJ<:Real}
    _check_symmetric(J)
    n = size(J, 1)
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _bc_matrix(J, crystal, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_matrix(J, crystal, float(h))
    else
        _bc_matrix(J, crystal, NoField())
    end
end

function _bc_matrix(J, crystal, h)
    n = size(J, 1)
    sys = BlumeCapelMatrix{eltype(J), typeof(crystal), typeof(h)}(
        ones(Int8, n), J, crystal, h, 0.0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function _local_coupling(sys::BlumeCapelMatrix, i)
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i+1]-1)
        j = sys.J.rowval[ptr]
        j != i && (acc += sys.J.nzval[ptr] * sys.spins[j])
    end
    return acc
end

@inline function local_pair_interactions(sys::BlumeCapelMatrix, i)
    s_i = sys.spins[i]
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i+1]-1)
        j = sys.J.rowval[ptr]
        j != i && (acc += sys.J.nzval[ptr] * sys.spins[j])
    end
    return s_i * acc
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapel factory constructors
# ═══════════════════════════════════════════════════════════════════════════════

function BlumeCapel(dims::AbstractVector{<:Integer}; J=1, D=0, periodic=true, h=0)
    if J isa Real && periodic
        return BlumeCapelLattice(dims, J, D; h=h)
    end
    graph = Graphs.SimpleGraphs.grid(collect(Int, dims); periodic)
    if J isa Real
        return BlumeCapelGraph(graph, J, D; h=h)
    elseif J isa AbstractVector{<:Real}
        return BlumeCapel(graph, J, D; h=h)
    else
        error("J must be Real or AbstractVector{<:Real}")
    end
end

BlumeCapel(graph::SimpleGraph, J::Real, D::Real; h=0) = BlumeCapelGraph(graph, J, D; h=h)
BlumeCapel(J::SparseMatrixCSC, D::Real; h=0) = BlumeCapelMatrix(J, D; h=h)

function BlumeCapel(graph::SimpleGraph, J::AbstractVector{<:Real}, D::Real; h=0)
    @assert ne(graph) == length(J) "Length of J vector must equal number of graph edges"
    TJ = float(eltype(J))
    rows = Vector{Int}(undef, 2ne(graph))
    cols = Vector{Int}(undef, 2ne(graph))
    vals = Vector{TJ}(undef, 2ne(graph))
    k = 1
    for (idx, e) in enumerate(edges(graph))
        i, j = src(e), dst(e)
        Jij = TJ(J[idx])
        rows[k] = i; cols[k] = j; vals[k] = Jij; k += 1
        rows[k] = j; cols[k] = i; vals[k] = Jij; k += 1
    end
    n = nv(graph)
    return BlumeCapelMatrix(sparse(rows, cols, vals, n, n), D; h=h)
end
